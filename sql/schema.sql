-- Decision-facing tables only. Hidden simulation truth is NEVER stored here. ALL DATA IS SYNTHETIC.
-- Dates are ISO text 'YYYY-MM-DD'. A value 'reported_at' means: visible from 00:00 of that date.

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Grain: one row per installed (attributed) player. Includes players who never return.
CREATE TABLE players (
    player_id    INTEGER PRIMARY KEY,
    channel      TEXT NOT NULL,
    install_date TEXT NOT NULL,
    reported_at  TEXT NOT NULL,          -- when the install attribution became visible
    CHECK (reported_at > install_date)
);
CREATE INDEX idx_players_cohort ON players (channel, install_date);

-- Grain: one row per player per ACTIVE day (no row = inactive that day). Primary key makes joins 1:1 per player-day.
CREATE TABLE player_days (
    player_id      INTEGER NOT NULL REFERENCES players (player_id),
    event_date     TEXT    NOT NULL,
    day_n          INTEGER NOT NULL CHECK (day_n >= 0),   -- event_date minus the player's install_date
    sessions       INTEGER NOT NULL CHECK (sessions >= 1),
    levels         INTEGER NOT NULL CHECK (levels >= 0),
    ad_impressions INTEGER NOT NULL CHECK (ad_impressions >= 0),
    iap_gross      REAL    NOT NULL CHECK (iap_gross >= 0),
    ad_revenue     REAL    NOT NULL CHECK (ad_revenue >= 0),
    reported_at    TEXT    NOT NULL,
    PRIMARY KEY (player_id, event_date),
    CHECK (reported_at > event_date)
) WITHOUT ROWID;

-- Grain: one row per paid channel per calendar day. Organic has no spend row.
CREATE TABLE spend_daily (
    channel     TEXT NOT NULL,
    spend_date  TEXT NOT NULL,
    spend       REAL NOT NULL CHECK (spend >= 0),
    reported_at TEXT NOT NULL,
    PRIMARY KEY (channel, spend_date),
    CHECK (reported_at > spend_date)
);

-- Grain: one row per cohort (channel x install_date) per milestone. complete_at is the declared date from which all
-- rows for install_date .. install_date + through_day are final (install + through_day + the channel's max lag).
CREATE TABLE cohort_reporting (
    channel     TEXT NOT NULL,
    install_date TEXT NOT NULL,
    through_day INTEGER NOT NULL CHECK (through_day IN (3, 30, 180)),
    complete_at TEXT NOT NULL,
    PRIMARY KEY (channel, install_date, through_day)
);
