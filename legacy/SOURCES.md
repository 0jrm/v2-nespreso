# Legacy source provenance

| Source | Ported into | Left behind |
|--------|-------------|-------------|
| `GEM_SubsurfaceFields/singleFileModel_SAT_stats4verticalProj_meeting20260203.py` | Repo root monolith → future `src/nespreso/*` phases | — |
| `2025-2_OCP-project/metrics.py` | `src/nespreso/physics_metrics.py` | — |
| `GEM_SubsurfaceFields/eoas_pyutils/io_utils/coaps_io_data.py` | `src/nespreso/io/satellite_readers.py` | Full eoas_pyutils tree |
| `GEM_SubsurfaceFields/eoas_pyutils/download_data/Download_Aviso_SSH_UV.py` | `src/nespreso/io/download/aviso.py` | motuclient shell wrapper |
| `GEM_SubsurfaceFields/eoas_pyutils/download_data/Download_SST_OISST.py` | Replaced by `io/download/copernicus.py` (`download_ostia_sst`) | Dead PODAAC subscriber |
| `NeSPReSO2/utils/copernicus_data_download.py` | `io/download/copernicus.py` | Hardcoded credentials |
| `global_nespreso/old/data_extract/` | Planned `src/nespreso/ingest/` (phase infra-port) | — |
| NeSPReSO2 autoencoder/KAN/`token_transformer.py` | Indexed only | Not merged (reference) |
