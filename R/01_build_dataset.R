# 01_build_dataset.R
# Build a resolved-market feature table from a locally obtained hourly
# Polymarket Parquet file and the public Gamma / CLOB APIs.
#
# Rscript R/01_build_dataset.R path/to/events-2026-05-30T21.parquet
#
# Raw hourly Parquet data was made available for this course under restricted
# access. This script intentionally does NOT include credentials or download
# protected datasets. Only publish data you have permission to redistribute.

required_packages <- c("arrow", "dplyr", "jsonlite")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Install required packages: install.packages(c('",
       paste(missing_packages, collapse = "', '"), "'))")
}

library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
parquet_path <- if (length(args)) args[[1]] else "data/raw/events-2026-05-30T21.parquet"
if (!file.exists(parquet_path)) {
  stop("Hourly Parquet file not found: ", parquet_path,
       "\nPass the path as the first command-line argument.")
}

dir.create("data/cache", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

# Fixed historical forecast date: 2026-05-30 14:00 PDT = 21:00 UTC.
cutoff_time <- as.POSIXct("2026-05-30 21:00:00", format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
cutoff <- as.numeric(cutoff_time)
lookback_seconds <- 48 * 3600

# The hourly source is ~21 million order-book updates; we need only the IDs.
market_ids <- unique(arrow::read_parquet(parquet_path, col_select = "market")$market)
cat("Unique market IDs:", length(market_ids), "\n")

# JSON arrays in Gamma metadata are sometimes encoded as character strings.
decode_array <- function(x) {
  if (is.null(x) || length(x) == 0) return(character(0))
  if (is.character(x) && length(x) == 1 && grepl("^\\s*\\[", x)) {
    x <- tryCatch(jsonlite::fromJSON(x), error = function(e) character(0))
  }
  unname(unlist(x, recursive = TRUE, use.names = FALSE))
}

parse_api_time <- function(x) {
  if (is.null(x) || length(x) != 1 || is.na(x) || !nzchar(x)) {
    return(as.POSIXct(NA, tz = "UTC"))
  }
  # Handles a leading YYYY-MM-DDTHH:MM:SS component, with or without fractions.
  as.POSIXct(substr(x, 1, 19), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
}

# Failures are retried, then recorded as missing. The cache prevents unnecessary
# repeat requests and preserves the metadata used to create each cohort.
request_json <- function(url, tries = 3) {
  for (attempt in seq_len(tries)) {
    response <- tryCatch(jsonlite::fromJSON(url, simplifyVector = FALSE),
                         error = function(e) NULL)
    if (!is.null(response)) return(response)
    Sys.sleep(attempt)
  }
  NULL
}

# ---------------------------------------------------------------------------
# 1. Market metadata and candidate resolution labels
# ---------------------------------------------------------------------------
metadata_batches <- split(market_ids, ceiling(seq_along(market_ids) / 20))
all_closed <- list()
failed_metadata_batches <- integer(0)

for (i in seq_along(metadata_batches)) {
  cache_file <- sprintf("data/cache/gamma_%03d.rds", i)
  if (file.exists(cache_file)) {
    batch_result <- readRDS(cache_file)
  } else {
    ids <- metadata_batches[[i]]
    query <- paste0("condition_ids=", ids, collapse = "&")
    url <- paste0("https://gamma-api.polymarket.com/markets?closed=true&limit=100&", query)
    batch_result <- request_json(url)
    if (!is.null(batch_result)) saveRDS(batch_result, cache_file)
    Sys.sleep(0.25)
  }
  if (is.null(batch_result)) {
    failed_metadata_batches <- c(failed_metadata_batches, i)
  } else {
    all_closed <- c(all_closed, batch_result)
  }
}

if (length(failed_metadata_batches)) {
  warning("Metadata batches failed: ", paste(failed_metadata_batches, collapse = ", "),
          ". Re-run the script; successful batches are cached.")
}
if (!length(all_closed)) stop("No closed market metadata retrieved.")

# Deduplicate by market ID. Public API fields reflect their retrieval date;
# saving the metadata cache is important for reproducibility.
condition_ids <- vapply(all_closed, function(m) {
  if (is.null(m$conditionId)) NA_character_ else as.character(m$conditionId)
}, character(1))
valid <- !is.na(condition_ids) & !duplicated(condition_ids)
all_closed <- all_closed[valid]


get_event_group <- function(m) {
  if (!is.null(m$events) && length(m$events) &&
      !is.null(m$events[[1]]$id)) {
    return(paste0("event_", m$events[[1]]$id))
  }
  # This fallback cannot guarantee related markets stay in one group.
  paste0("market_", m$conditionId)
}

get_market_record <- function(m) {
  outcomes <- as.character(decode_array(m$outcomes))
  tokens <- as.character(decode_array(m$clobTokenIds))
  final_prices <- suppressWarnings(as.numeric(decode_array(m$outcomePrices)))
  binary <- length(outcomes) == 2 && setequal(outcomes, c("Yes", "No")) &&
    length(tokens) == 2
  final_label <- length(final_prices) == 2 &&
    all(is.finite(final_prices)) && all(final_prices %in% c(0, 1)) &&
    sum(final_prices == 1) == 1
  closed_time <- parse_api_time(m$closedTime)
  scheduled_end <- parse_api_time(m$endDate)
  status <- if (is.null(m$umaResolutionStatus)) NA_character_ else
    as.character(m$umaResolutionStatus)
  is_resolved <- identical(tolower(status), "resolved")
  data.frame(
    market = as.character(m$conditionId),
    event_group = get_event_group(m),
    question = if (is.null(m$question)) "" else as.character(m$question),
    yes_token = if (binary) tokens[match("Yes", outcomes)] else NA_character_,
    outcome = if (binary && final_label) as.integer(final_prices[match("Yes", outcomes)] == 1) else NA_integer_,
    binary = binary,
    final_label = final_label,
    is_resolved = is_resolved,
    closed_after_cutoff = !is.na(closed_time) && closed_time > cutoff_time,
    end_after_cutoff = !is.na(scheduled_end) && scheduled_end > cutoff_time,
    stringsAsFactors = FALSE
  )
}

catalog <- dplyr::bind_rows(lapply(all_closed, get_market_record))
write.csv(catalog, "data/cache/market_catalog.csv", row.names = FALSE)

eligible <- catalog %>%
  filter(binary, final_label, is_resolved,
         closed_after_cutoff, end_after_cutoff)

cat("Closed markets retrieved:", nrow(catalog), "\n")
cat("Eligible candidate markets:", nrow(eligible), "\n")
cat("Outcome breakdown:\n")
print(table(eligible$outcome))

# WARNING: A market closing after the forecast cutoff does not prove that the
# underlying event was undecided at the cutoff. Manual/event-time validation
# remains necessary before claiming genuine predictive outperformance.

# ---------------------------------------------------------------------------
# 2. Historical CLOB price features for each eligible YES token
# ---------------------------------------------------------------------------
fetch_history_features <- function(record) {
  p_url <- paste0(
    "https://clob.polymarket.com/prices-history?market=",
    record$yes_token,
    "&startTs=", sprintf("%.0f", cutoff - lookback_seconds),
    "&endTs=", sprintf("%.0f", cutoff),
    "&fidelity=5"
  )
  result <- request_json(p_url)
  if (is.null(result) || is.null(result$history) ||
      length(result$history) < 3) return(NULL)

  # simplifyVector=FALSE gives a list of {t,p} points.
  h <- dplyr::bind_rows(lapply(result$history, function(z) {
    data.frame(t = suppressWarnings(as.numeric(z$t)),
               p = suppressWarnings(as.numeric(z$p)))
  }))
  h <- h %>%
    filter(is.finite(t), is.finite(p), p >= 0, p <= 1,
           t >= cutoff - lookback_seconds, t <= cutoff) %>%
    arrange(t) %>%
    distinct(t, .keep_all = TRUE)
  if (nrow(h) < 3) return(NULL)

  data.frame(
    market = record$market,
    event_group = record$event_group,
    question = record$question,
    probability = dplyr::last(h$p),
    price_dispersion = sd(h$p),
    volatility = sd(diff(h$p)),
    momentum = dplyr::last(h$p) - dplyr::first(h$p),
    observations = nrow(h),
    history_hours = (dplyr::last(h$t) - dplyr::first(h$t)) / 3600,
    minutes_since_price = (cutoff - dplyr::last(h$t)) / 60,
    outcome = record$outcome,
    stringsAsFactors = FALSE
  )
}

feature_batches <- split(seq_len(nrow(eligible)),
                         ceiling(seq_len(nrow(eligible)) / 25))
missing_histories <- character(0)
for (i in seq_along(feature_batches)) {
  f <- sprintf("data/cache/features_%03d.rds", i)
  if (file.exists(f)) next
  rows <- vector("list", length(feature_batches[[i]]))
  for (j in seq_along(feature_batches[[i]])) {
    index <- feature_batches[[i]][j]
    rec <- eligible[index, , drop = FALSE]
    rows[[j]] <- tryCatch(fetch_history_features(rec), error = function(e) {
      message("History request failed for ", rec$market, ": ", conditionMessage(e))
      NULL
    })
    if (is.null(rows[[j]])) missing_histories <- c(missing_histories, rec$market)
    Sys.sleep(0.2)
  }
  saveRDS(dplyr::bind_rows(rows), f)
  cat("Saved feature batch", i, "of", length(feature_batches), "\n")
}

files <- list.files("data/cache", pattern = "^features_[0-9]{3}\\.rds$", full.names = TRUE)
if (length(files) != length(feature_batches)) {
  warning("Expected ", length(feature_batches), " feature batches but found ", length(files),
          "; dataset may be incomplete.")
}
full_data <- dplyr::bind_rows(lapply(files, readRDS)) %>%
  distinct(market, .keep_all = TRUE)

analysis_data <- full_data %>%
  filter(history_hours >= 40,
         minutes_since_price <= 10,
         is.finite(probability),
         is.finite(volatility),
         is.finite(price_dispersion),
         is.finite(momentum),
         between(probability, 0, 1),
         outcome %in% c(0, 1))

model_data <- analysis_data %>%
  mutate(
    logit_market = qlogis(pmin(pmax(probability, 0.0001), 0.9999)),
    volatility_pp = volatility * 100,
    momentum_pp = momentum * 100,
    dispersion_pp = price_dispersion * 100
  )

# These outputs are intentionally local unless sharing is explicitly permitted.
write.csv(model_data, "data/processed/model_data.csv", row.names = FALSE)
saveRDS(model_data, "data/processed/model_data.rds")
cat("Rows in unfiltered feature table:", nrow(full_data), "\n")
cat("Rows in final model_data:", nrow(model_data), "\n")
cat("Event-group IDs available:", sum(startsWith(model_data$event_group, "event_")),
    "of", nrow(model_data), "\n")
if (length(missing_histories)) {
  warning(length(missing_histories), " price histories were unavailable on the first pass. ",
          "Review data/cache before interpreting any sample size.")
}
