-- Classic retention curve for one channel: for each day n = 0..30, share of ALL installed players of the matured cohorts
-- (cohorts whose D30 milestone is complete as of temp._as_of) who have a row on exactly day n. Reads ONLY v_* views.
-- Always returns 31 rows (D0..D30). A day with no activity rows has active_players = 0 and retention = 0.
-- If there are NO eligible installed players (no matured cohort, or none of its players visible yet), installs = 0,
-- retention is NULL and retention_available = 0: retention is unavailable, not zero.
WITH RECURSIVE days AS (
    SELECT 0 AS day_n
    UNION ALL
    SELECT day_n + 1 FROM days WHERE day_n < 30
),
matured AS (
    SELECT channel, install_date FROM v_cohort_reporting WHERE through_day = 30 AND channel = :channel
),
denominator AS (
    SELECT COUNT(*) AS installs
    FROM v_players p JOIN matured m ON m.channel = p.channel AND m.install_date = p.install_date
),
numerator AS (
    SELECT d.day_n, COUNT(*) AS active_players
    FROM v_player_days d JOIN matured m ON m.channel = d.channel AND m.install_date = d.install_date
    WHERE d.day_n <= 30
    GROUP BY d.day_n
)
SELECT days.day_n,
       COALESCE(numerator.active_players, 0) AS active_players,
       denominator.installs,
       CASE WHEN denominator.installs > 0
            THEN COALESCE(numerator.active_players, 0) * 1.0 / denominator.installs END AS retention,
       CASE WHEN denominator.installs > 0 THEN 1 ELSE 0 END AS retention_available
FROM days
CROSS JOIN denominator
LEFT JOIN numerator ON numerator.day_n = days.day_n
ORDER BY days.day_n;
