# Billboard Top 100 — Cleaned Dataset Field Reference

## Song metadata (Genius)

| Column | Type | Description |
|---|---|---|
| `song_title` | str | Song title |
| `artist` | str | Primary artist |
| `featured_artists` | str | Featured artists (JSON list string, e.g. `['Drake']` or `[]`) |
| `album` | str | Album name |
| `album_url` | str | Genius album page URL |
| `song_url` | str | Genius song page URL |
| `release_date` | str | Release date — Genius value preferred, Spotify `album_release_date` as fallback |
| `writers` | str | Songwriters (JSON list string) |

## Chart metadata (Billboard)

| Column | Type | Description |
|---|---|---|
| `year` | int | Billboard chart year |
| `rank` | int | Chart rank (1–100) |

## Lyrics (Genius)

| Column | Type | Description |
|---|---|---|
| `lyrics_raw` | str | Original scraped lyrics including `[Verse 1]`, `[Chorus]` etc. section markers |
| `lyrics` | str | Cleaned lyrics — section markers stripped, excess whitespace normalized |

## Spotify metadata

| Column | Type | Description |
|---|---|---|
| `id` | str | Spotify track ID |
| `spotify_url` | str | Spotify track URL |
| `explicit` | bool | Explicit content flag |
| `duration_ms` | float | Track duration in milliseconds |
| `popularity` | float | Spotify popularity score (0–100) |
| `total_artist_followers` | float | Total follower count across all credited artists |
| `avg_artist_popularity` | float | Average Spotify popularity score across all credited artists |
| `artist_ids` | str | Spotify artist IDs (JSON list string) |

## Genre (Spotify)

| Column | Type | Description |
|---|---|---|
| `genre` | str | Broad genre label (e.g. `Pop`, `Rock`, `R&B`) — ~45% coverage |
| `niche_genres` | str | Fine-grained genre tags from Spotify (JSON list string, e.g. `['classic country', 'honky tonk']`) |

## Audio features (Spotify)

All values are floats. See [Spotify audio features docs](https://developer.spotify.com/documentation/web-api/reference/get-audio-features) for full definitions.

| Column | Range | Description |
|---|---|---|
| `danceability` | 0–1 | How suitable the track is for dancing |
| `energy` | 0–1 | Perceptual intensity and activity |
| `key` | 0–11 | Musical key (0 = C, 1 = C♯/D♭, … 11 = B) |
| `loudness` | dB | Overall loudness (typically −60 to 0) |
| `mode` | 0 or 1 | Modality: 1 = major, 0 = minor |
| `speechiness` | 0–1 | Presence of spoken words (>0.66 = spoken, <0.33 = music) |
| `acousticness` | 0–1 | Confidence the track is acoustic |
| `instrumentalness` | 0–1 | Predicts whether a track has no vocals (>0.5 = likely instrumental) |
| `liveness` | 0–1 | Presence of a live audience (>0.8 = likely live recording) |
| `valence` | 0–1 | Musical positivity — high = happy/euphoric, low = sad/angry |
| `tempo` | BPM | Estimated tempo in beats per minute |
