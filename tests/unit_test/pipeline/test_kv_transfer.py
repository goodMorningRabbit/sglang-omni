    with capture_comm_trace(monkeypatch) as events:

        async def run() -> None:
            relay, source, destination = await start_pair()
            try:
                source.register_kv_pool(make_pool("source_pool"))
                destination.register_kv_pool(make_pool("destination_pool"))
                destination.register_kv_receiver(
                    "destination_pool", FailingReceiver((0,))
                )
                with pytest.raises(RuntimeError, match="rank-local reserve failed"):
                    await source.send_kv_pages(
                        request_id="request",
                        transfer_id="transfer",
                        source_pool_id="source_pool",
                        source_page_indices=(1,),
                        target_pool_id="destination_pool",
                        to_stage="destination",
                        lease=Mock(),
                    )
            finally:
                await source.close()
                await destination.close()

        asyncio.run(run())

    assert kv_events(events) == [
        "comm_kv_send_start",
        "comm_kv_prepare_rejected",
        "comm_kv_ready",
        "comm_kv_transfer_failed",
    ]

    # The receiver names the reason, so the sender does not have to guess it.
    rejected = first(events, "comm_kv_prepare_rejected")
    assert rejected["transfer_id"] == "transfer"
    assert rejected["target_pool_id"] == "destination_pool"
    assert "rank-local reserve failed" in rejected["error"]

    ready = first(events, "comm_kv_ready")
    assert ready["success"] is False
    assert "rank-local reserve failed" in ready["error"]


def test_kv_ack_timeout_traces_the_retained_transfer_with_a_running_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with capture_comm_trace(monkeypatch) as events:

        async def run() -> None:
            relay, source, destination = await start_pair()

            async def drop_data_ready(
                sockets: dict[str, Any], target_endpoint: str, message: Any
            ) -> None:
                if isinstance(message, DataReadyMessage):
                    return
                await send_to_endpoint(sockets, target_endpoint, message)

            monkeypatch.setattr(
                "sglang_omni.comm.engine.send_to_endpoint",
                drop_data_ready,
            )
            source.ack_timeout_s = 0.1
            source.register_kv_pool(make_pool("source_pool"))
            destination.register_kv_pool(make_pool("destination_pool"))
            destination.register_kv_receiver("destination_pool", Receiver((0,)))
            try:
                with pytest.raises(TimeoutError):
                    await source.send_kv_pages(
                        request_id="request",
                        transfer_id="transfer",
                        source_pool_id="source_pool",
                        source_page_indices=(1,),
                        target_pool_id="destination_pool",
                        to_stage="destination",
                        lease=Mock(),
                    )
                assert len(source.retained_pending_kv_transfers) == 1
            finally:
                await source.close()
                await destination.close()

        asyncio.run(run())

    retained = first(events, "comm_kv_pending_retained")
    assert retained["object_id"] == "transfer"
    assert retained["retained_count"] == 1
    assert retained["num_ops"] == 1
    assert retained["bytes"] == 1
    failed = first(events, "comm_transfer_failed")
    assert failed["object_id"] == "transfer"
    assert failed["num_ops"] == 1
    assert failed["error"] == "TimeoutError"
    assert failed["timeout_s"] == 0.1
    assert failed["elapsed_ms"] >= 0.0
    assert "comm_kv_transfer_failed" in kv_events(events)


def test_kv_transfer_emits_nothing_when_the_env_gate_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with capture_comm_trace(monkeypatch, enable=False) as events:

        async def run() -> None:
            relay, source, destination = await start_pair()
            try:
                source.register_kv_pool(make_pool("source_pool"))
                destination.register_kv_pool(make_pool("destination_pool"))
                destination.register_kv_receiver("destination_pool", Receiver((0, 3)))
                await source.send_kv_pages(
                    request_id="request",
                    transfer_id="transfer",
                    source_pool_id="source_pool",
                    source_page_indices=(1, 4),
                    target_pool_id="destination_pool",
                    to_stage="destination",
                    lease=Mock(),
                )
            finally:
                await source.close()
                await destination.close()

        asyncio.run(run())

    assert events == []
