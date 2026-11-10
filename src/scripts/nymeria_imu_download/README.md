# Download Nymeria

Generate a download catalogue through the [Aria Dataset Explorer](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/dataset_explorer). Its links expire and may contain access tokens, so the catalogue is not bundled. Keep it locally, outside version control.

Download the modalities used by this repository with:

```sh
export NYMERIA_JSON="/path/to/local/catalogue.json"
export FOUNDATIONIMU_DATA_ROOT="$PWD/data"
bash src/scripts/nymeria_imu_download/download_nymeria.sh
```

The shell downloader requires `jq`, `curl`, `unzip`, `sha1sum`, `awk`, and `flock`. Set `NYMERIA_DRY_RUN=1` to list the work, `NYMERIA_MAX_SEQUENCES` to limit it, and `NYMERIA_PARALLEL` to control concurrency. For a small subset:

```sh
python -m src.scripts.nymeria_imu_download.download_subset --manifest "$NYMERIA_JSON" --sequences 10
```

Both default to `$FOUNDATIONIMU_DATA_ROOT/nymeria_imu`. Preserve the official sequence IDs when downloading: the committed split manifests refer to them. Name-like fields in these public IDs are labelled `fake_name` by the dataset publisher ([example official record](https://explorer.projectaria.com/nymeria/20231222_s0_denise_carter_act0_3rt2lg)); displayed trajectory labels use the opaque take UID.

The following illustrates the expected shape only; it is not a working catalogue:

```json
{
  "sequences": {
    "official-sequence-id": {
      "body_motion": {
        "download_url": "https://example.invalid/local-only-download",
        "filename": "body_motion.zip",
        "file_size_bytes": 0,
        "sha1sum": "sha1-from-your-catalogue"
      }
    }
  }
}
```
