# Real-music subset: sources and license

- **Audio**: ISMIR 2004 Tempo Induction Contest "Ballroom" dataset, 698 ~30s
  excerpts, hosted at
  `https://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz`
  (Music Technology Group, Universitat Pompeu Fabra; Gouyon et al., "An
  experimental comparison of audio tempo induction algorithms", IEEE TASLP
  14(5), 2006). Distributed for research/benchmarking use by MTG-UPF; no
  login required.
- **Beat/downbeat annotations**: CPJKU `BallroomAnnotations` repository,
  `https://github.com/CPJKU/BallroomAnnotations` (MIT-style permissive,
  see repo). One `.beats` file per track: `<time_seconds> <beat_index_in_bar>`,
  where index `1` marks the downbeat.
- **Sampling**: stratified by dance-style folder (proportional to each
  folder's count of annotated files), `random.seed(42)`, ~40 files
  requested -> 41 selected across 9 folders (Jive, Quickstep, Tango, Waltz,
  VienneseWaltz, Samba, ChaChaCha, Rumba-International, Rumba-Misc).
  454/698 files in the audio set have matching annotations; only those were
  eligible.
- **Schema**: `M/real/truth/<Genre>_<Media-id>.json` follows the same schema
  as the synthetic truth files (`beats`, `downbeats`, `meter`, `tempo_bpm`,
  `subset`). `meter` and `tempo_bpm` here are *derived* from the human
  annotations (median beats-per-bar between downbeats; median beat-interval
  BPM) since the original contest did not publish a meter field -- this is
  the one subset where truth is not from generation parameters, by nature
  of being real recorded music with human annotations.
- Audio in `M/real/audio/*.wav`, one file per selected track (renamed
  `<Genre>_<Media-id>.wav`).
