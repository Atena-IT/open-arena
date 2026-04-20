"""Open Arena package."""

import warnings

# Transitive deps (polyfile-weave → chardet 7.x) pull in a chardet newer than
# requests' declared compat range. requests prefers charset_normalizer at
# runtime, so suppress only that specific requests-emitted warning rather than
# globally hiding urllib3/chardet version-mismatch warnings for the process.
warnings.filterwarnings(
    "ignore",
    message=r".*chardet.*doesn't match a supported version.*",
    module=r"^requests(?:\.|$)",
)
