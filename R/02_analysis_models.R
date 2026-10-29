# 02_analysis_models.R
# Analyze model_data.csv and compare Polymarket prices with Logistic Regression
# and Random Forest on an event-grouped holdout sample.
#
# Run from the repository root: Rscript R/02_analysis_models.R

required_packages <- c("dplyr", "ggplot2", "ranger")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Install required packages: install.packages(c('",
       paste(missing_packages, collapse = "', '"), "'))")
}

library(dplyr)
library(ggplot2)
library(ranger)

dir.create("results", recursive = TRUE, showWarnings = FALSE)
input <- "data/processed/model_data.csv"
if (!file.exists(input)) stop("Missing ", input, ". First run 01_build_dataset.R ",
                              "or copy your exported model_data.csv there.")
model_data <- read.csv(input, stringsAsFactors = FALSE)

necessary <- c("market", "event_group", "probability", "volatility",
               "momentum", "price_dispersion", "outcome")
if (!all(necessary %in% names(model_data))) {
  stop("Missing columns: ", paste(setdiff(necessary, names(model_data)), collapse = ", "))
}

# Reproduce the feature transformations in the original R Markdown, rather than
# trusting any transforms saved in a pre-existing CSV.
model_data <- model_data %>%
  filter(is.finite(probability), is.finite(volatility),
         is.finite(momentum), is.finite(price_dispersion),
         between(probability, 0, 1), outcome %in% c(0, 1)) %>%
  mutate(
    event_group = ifelse(is.na(event_group) | event_group == "",
                         paste0("market_", market), event_group),
    logit_market = qlogis(pmin(pmax(probability, 0.0001), 0.9999)),
    volatility_pp = 100 * volatility,
    momentum_pp = 100 * momentum,
    dispersion_pp = 100 * price_dispersion
  )

if (!nrow(model_data)) stop("No eligible observations after data validation.")
if (length(unique(model_data$outcome)) < 2) stop("Need both YES and NO labels.")

brier <- function(p, y) mean((p - y)^2)

# ---------------------------------------------------------------------------
# Descriptive market performance on the full analysis cohort
# ---------------------------------------------------------------------------
base_rate <- mean(model_data$outcome)
baseline_results <- data.frame(
  n_markets = nrow(model_data),
  observed_yes_rate = base_rate,
  market_brier = brier(model_data$probability, model_data$outcome),
  constant_brier_in_sample = brier(rep(base_rate, nrow(model_data)), model_data$outcome),
  always_no_brier = brier(rep(0, nrow(model_data)), model_data$outcome),
  market_accuracy = mean(as.integer(model_data$probability >= 0.5) == model_data$outcome),
  always_no_accuracy = mean(model_data$outcome == 0)
)
write.csv(baseline_results, "results/descriptive_baselines.csv", row.names = FALSE)
print(baseline_results)

calibration <- model_data %>%
  mutate(price_bin = cut(probability, breaks = seq(0, 1, 0.1), include.lowest = TRUE)) %>%
  group_by(price_bin) %>%
  summarise(n = n(), predicted = mean(probability), actual = mean(outcome),
            .groups = "drop")
write.csv(calibration, "results/calibration_bins.csv", row.names = FALSE)

plot_calibration <- ggplot(calibration, aes(predicted, actual)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(color = "#00A99B") +
  geom_point(aes(size = n), color = "#0B375B") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = "Polymarket forecast calibration",
       subtitle = sprintf("N = %s markets; prediction cutoff: May 30, 2026", nrow(model_data)),
       x = "Mean historical YES probability", y = "Observed YES frequency",
       size = "Markets") +
  theme_minimal(base_size = 12)
ggsave("results/calibration_plot.png", plot_calibration, width = 8.5, height = 5.5, dpi = 200)

# ---------------------------------------------------------------------------
# One event-grouped holdout split; NOT a chronological out-of-sample backtest.
# ---------------------------------------------------------------------------
set.seed(42)
groups <- unique(model_data$event_group)
train_groups <- sample(groups, size = floor(0.8 * length(groups)))
train <- model_data %>% filter(event_group %in% train_groups)
test <- model_data %>% filter(!event_group %in% train_groups)

if (length(unique(train$outcome)) < 2) stop("Training split has only one outcome.")
if (!nrow(test)) stop("Test split is empty.")
stopifnot(!any(train$event_group %in% test$event_group))
cat("Train/test rows:", nrow(train), "/", nrow(test), "\n")

model_price <- glm(outcome ~ logit_market, data = train, family = binomial())
model_full <- glm(outcome ~ logit_market + volatility_pp + momentum_pp +
                    dispersion_pp, data = train, family = binomial())

train_rf <- train %>% mutate(outcome_rf = factor(outcome, levels = c(0, 1)))
rf_model <- ranger(
  outcome_rf ~ logit_market + volatility_pp + momentum_pp + dispersion_pp,
  data = train_rf, probability = TRUE, num.trees = 500,
  min.node.size = 20, seed = 42
)
rf_predictions <- predict(rf_model, data = test)$predictions

# Test predictions all refer to the same held-out markets.
test <- test %>%
  mutate(
    pred_market = probability,
    pred_price = predict(model_price, newdata = test, type = "response"),
    pred_full = predict(model_full, newdata = test, type = "response"),
    pred_rf = rf_predictions[, "1"]
  )

# Out-of-sample constant probability estimated exclusively on training labels.
constant_test <- rep(mean(train$outcome), nrow(test))
comparison <- data.frame(
  model = c("Polymarket", "Constant: Train YES Rate", "Logistic: Price Only",
            "Logistic: Full", "Random Forest"),
  brier = c(brier(test$pred_market, test$outcome),
            brier(constant_test, test$outcome),
            brier(test$pred_price, test$outcome),
            brier(test$pred_full, test$outcome),
            brier(test$pred_rf, test$outcome))
) %>% arrange(brier)

write.csv(comparison, "results/model_comparison.csv", row.names = FALSE)
write.csv(train, "results/train.csv", row.names = FALSE)
write.csv(test, "results/test_predictions.csv", row.names = FALSE)
print(comparison)

plot_models <- ggplot(comparison, aes(x = reorder(model, brier), y = brier)) +
  geom_col(fill = "#0B375B", width = 0.65) +
  coord_flip() +
  labs(title = "Held-out Brier scores", subtitle = "Lower is better; one event-grouped holdout split",
       x = NULL, y = "Brier score") +
  theme_minimal(base_size = 12)
ggsave("results/model_brier_comparison.png", plot_models, width = 8.5, height = 5, dpi = 200)

market_score <- comparison$brier[comparison$model == "Polymarket"]
rf_score <- comparison$brier[comparison$model == "Random Forest"]
cat(sprintf("Random Forest relative Brier change vs. market: %.2f%%\n",
            100 * (rf_score - market_score) / market_score))
cat("NOTE: A difference on one grouped split does not establish a robust predictive edge.\n")
