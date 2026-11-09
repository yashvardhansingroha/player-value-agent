-- Matured cohort outcome: cumulative net revenue over day 0..:through_day (inclusive), only for cohorts whose
-- :through_day milestone is declared complete as of temp._as_of. Reads ONLY v_* views.
-- :through_day must be 3, 30 or 180 (checked in Python). :fee is the store fee.
WITH ready AS (
    SELECT channel, install_date FROM v_cohort_reporting WHERE through_day = :through_day
),
size AS (
    SELECT channel, install_date, COUNT(*) AS installs FROM v_players GROUP BY channel, install_date
),
spend AS (
    SELECT channel, spend_date AS install_date, spend FROM v_spend
),
rev AS (
    SELECT channel, install_date,
           SUM(iap_gross)  AS gross_iap,
           SUM(ad_revenue) AS ad_revenue
    FROM v_player_days
    WHERE day_n <= :through_day
    GROUP BY channel, install_date
)
SELECT r.channel,
       r.install_date,
       COALESCE(size.installs, 0)                                          AS installs,
       spend.spend                                                         AS spend,
       COALESCE(rev.gross_iap, 0)                                          AS gross_iap,
       COALESCE(rev.ad_revenue, 0)                                         AS ad_revenue,
       COALESCE(rev.gross_iap, 0) * (1 - :fee) + COALESCE(rev.ad_revenue, 0) AS net_revenue
FROM ready r
LEFT JOIN size  ON size.channel  = r.channel AND size.install_date  = r.install_date
LEFT JOIN spend ON spend.channel = r.channel AND spend.install_date = r.install_date
LEFT JOIN rev   ON rev.channel   = r.channel AND rev.install_date   = r.install_date
ORDER BY r.channel, r.install_date;
