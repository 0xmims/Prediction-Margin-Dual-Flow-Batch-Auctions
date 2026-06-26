# Future Empirical Data Requirements

The synthetic scaffold should eventually be paired with empirical replay. Message-level data is ideal because stale-quote races are message-order phenomena.

## Kalshi

Future Kalshi data needs include:

- market metadata;
- L2 order book snapshots;
- L2 deltas;
- public trades;
- taker side;
- order lifecycle data, if permissioned;
- fills, if permissioned;
- trading status;
- event and news labels.

## Polymarket

Future Polymarket data needs include:

- Gamma/event metadata;
- token mapping;
- CLOB order book snapshots;
- CLOB deltas or websocket updates;
- on-chain `OrderFilled` logs;
- market resolution;
- event and news labels.

## Identification Limits

Snapshots and trades can estimate stale-loss proxies, such as post-news adverse fills near stale prices, but they cannot fully identify failed cancels, failed orders, or true latency races.

Without order lifecycle data and cancel messages, we cannot claim to observe true maker-cancel versus taker-hit races. Any empirical work with only snapshots and trades should be framed as proxy measurement.
