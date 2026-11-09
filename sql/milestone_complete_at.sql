-- Declared completeness date of a milestone (:through_day = 3, 30 or 180) for cohorts whose milestone is complete as of
-- temp._as_of. Reads ONLY v_* views.
SELECT channel, install_date, complete_at
FROM v_cohort_reporting
WHERE through_day = :through_day
ORDER BY channel, install_date;
