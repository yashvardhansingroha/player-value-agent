-- Cohort metrics as of the date in temp._as_of. Reads ONLY v_* views (see availability.py).
-- Grain of the result: one row per cohort = channel x install_date. Every CTE is already at that grain
-- (GROUP BY channel, install_date) before it is joined, so a join cannot multiply spend or revenue.
-- Parameter :fee is the store fee (net IAP = gross IAP * (1 - :fee)).
-- D3 metrics are NULL until the cohort's D3 milestone is declared complete (d3_complete = 1).
-- Retention Dn = players with a row on exactly day n / ALL visible installed players (inactive players included).
WITH keys AS (
    SELECT channel, install_date FROM v_players GROUP BY channel, install_date
    UNION
    SELECT channel, spend_date AS install_date FROM v_spend
),
size AS (
    SELECT channel, install_date, COUNT(*) AS installs_reported
    FROM v_players GROUP BY channel, install_date
),
ready AS (
    SELECT channel, install_date, 1 AS d3_complete
    FROM v_cohort_reporting WHERE through_day = 3
),
spend AS (
    SELECT channel, spend_date AS install_date, spend FROM v_spend
),
d3 AS (
    SELECT channel, install_date,
           SUM(CASE WHEN day_n = 1 THEN 1 ELSE 0 END) AS active_d1,
           SUM(CASE WHEN day_n = 3 THEN 1 ELSE 0 END) AS active_d3,
           COUNT(*)                                    AS active_player_days,
           SUM(sessions)                               AS sessions,
           SUM(levels)                                 AS levels,
           SUM(iap_gross)                              AS iap_gross,
           SUM(ad_revenue)                             AS ad_revenue,
           COUNT(DISTINCT CASE WHEN iap_gross > 0 THEN player_id END) AS payers
    FROM v_player_days
    WHERE day_n <= 3
    GROUP BY channel, install_date
)
SELECT k.channel,
       k.install_date,
       COALESCE(size.installs_reported, 0)                       AS installs_reported,
       spend.spend                                               AS spend,
       COALESCE(ready.d3_complete, 0)                            AS d3_complete,
       CASE WHEN ready.d3_complete = 1 AND spend.spend IS NOT NULL AND size.installs_reported > 0
            THEN spend.spend * 1.0 / size.installs_reported END  AS cpi,
       CASE WHEN ready.d3_complete = 1 AND size.installs_reported > 0
            THEN COALESCE(d3.active_d1, 0) * 1.0 / size.installs_reported END AS d1_retention,
       CASE WHEN ready.d3_complete = 1 AND size.installs_reported > 0
            THEN COALESCE(d3.active_d3, 0) * 1.0 / size.installs_reported END AS d3_retention,
       CASE WHEN ready.d3_complete = 1
            THEN COALESCE(d3.iap_gross, 0) * (1 - :fee) + COALESCE(d3.ad_revenue, 0) END AS d0_3_net_revenue,
       CASE WHEN ready.d3_complete = 1 THEN COALESCE(d3.iap_gross, 0) END   AS d0_3_gross_iap,
       CASE WHEN ready.d3_complete = 1 THEN COALESCE(d3.ad_revenue, 0) END  AS d0_3_ad_revenue,
       CASE WHEN ready.d3_complete = 1 AND size.installs_reported > 0
            THEN COALESCE(d3.payers, 0) * 1.0 / size.installs_reported END  AS payer_rate_d0_3,
       CASE WHEN ready.d3_complete = 1 AND d3.active_player_days > 0
            THEN (d3.iap_gross * (1 - :fee) + d3.ad_revenue) / d3.active_player_days END AS arpdau_d0_3,
       CASE WHEN ready.d3_complete = 1 AND d3.active_player_days > 0
            THEN d3.sessions * 1.0 / d3.active_player_days END              AS sessions_per_active_user,
       CASE WHEN ready.d3_complete = 1 AND d3.active_player_days > 0
            THEN d3.levels * 1.0 / d3.active_player_days END                AS levels_per_active_user
FROM keys k
LEFT JOIN size  ON size.channel  = k.channel AND size.install_date  = k.install_date
LEFT JOIN ready ON ready.channel = k.channel AND ready.install_date = k.install_date
LEFT JOIN spend ON spend.channel = k.channel AND spend.install_date = k.install_date
LEFT JOIN d3    ON d3.channel    = k.channel AND d3.install_date    = k.install_date
ORDER BY k.channel, k.install_date;
