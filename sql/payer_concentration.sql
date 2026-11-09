-- D0-D3 gross IAP per paying player for cohorts of one channel installed between :from_date and :to_date (inclusive),
-- largest first. ELIGIBLE POPULATION: only cohorts whose D3 milestone is declared complete as of temp._as_of (the join to
-- v_cohort_reporting is 1:1 on its primary key, so it cannot multiply rows); partially reported cohorts are excluded, never
-- treated as complete. Reads ONLY v_* views.
SELECT d.player_id, SUM(d.iap_gross) AS iap_gross
FROM v_player_days d
JOIN v_cohort_reporting r
  ON r.channel = d.channel AND r.install_date = d.install_date AND r.through_day = 3
WHERE d.channel = :channel
  AND d.install_date BETWEEN :from_date AND :to_date
  AND d.day_n <= 3
GROUP BY d.player_id
HAVING SUM(d.iap_gross) > 0
ORDER BY iap_gross DESC;
