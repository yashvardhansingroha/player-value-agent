-- Per calendar day for one channel: active player-days reported that day, and the number of players installed in the
-- previous 30 days (inclusive). Driven by days with visible spend, so a day with ZERO activity rows still appears
-- (active_player_days = 0). Reads ONLY v_* views. Callers must ignore the most recent days whose events may still be
-- unreported (declared lag).
WITH act AS (
    SELECT event_date, COUNT(*) AS active_player_days
    FROM v_player_days
    WHERE channel = :channel
    GROUP BY event_date
),
inst AS (
    SELECT install_date, COUNT(*) AS n
    FROM v_players
    WHERE channel = :channel
    GROUP BY install_date
)
SELECT s.spend_date AS event_date,
       COALESCE(act.active_player_days, 0) AS active_player_days,
       (SELECT COALESCE(SUM(inst.n), 0) FROM inst
         WHERE inst.install_date BETWEEN date(s.spend_date, '-30 day') AND s.spend_date) AS recent_installs
FROM v_spend s
LEFT JOIN act ON act.event_date = s.spend_date
WHERE s.channel = :channel
ORDER BY s.spend_date;
