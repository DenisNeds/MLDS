library(dplyr)
library(ggplot2)
library(lme4)
library(patchwork)

df <- read.csv("Desktop/Stuff/DenisFaks/MLDS/HW2/predictions.csv")

df <- df %>% mutate(error = ifelse(y_true == y_pred, 0, 1)) 

df$angle_bin <- cut(df$Angle, breaks = seq(floor(min(df$Angle)),
                                           ceiling(max(df$Angle)),
                                           by = 5),
                    include.lowest = TRUE)
bin_summary <- df %>%
  group_by(Model, angle_bin) %>%
  summarize(
    angle_mid = mean(Angle, na.rm = TRUE),
    mean_error = mean(error, na.rm = TRUE),
    se_error = sd(error, na.rm = TRUE) / sqrt(n()),
    mean_uncertainty = mean(1 - prob_correct_class, na.rm = TRUE),
    se_uncertainty = sd(1 - prob_correct_class, na.rm = TRUE) / sqrt(n()),
    n = n(),
    .groups = "drop"
  )
print(bin_summary)

df_lr_bin <- bin_summary %>% filter(Model == "Logistic_Regression")

p_lr_err <- ggplot(df_lr_bin, aes(x = angle_mid, y = mean_error)) +
  geom_col(fill = "lightskyblue", color = "black") +
  geom_errorbar(aes(ymin = mean_error - 1.96 * se_error, ymax = mean_error + 1.96 * se_error), width = 2, color = "black", size = 0.8) +
  labs(title = "Actual Error", x = "Angle", y = "Mean Error") +
  theme_minimal()

p_lr_unc <- ggplot(df_lr_bin, aes(x = angle_mid, y = mean_uncertainty)) +
  geom_col(fill = "tomato", color = "black") +
  geom_errorbar(aes(ymin = mean_uncertainty - 1.96 * se_uncertainty, ymax = mean_uncertainty + 1.96 * se_uncertainty), width = 2, color = "black", size = 0.8) +
  labs(title = "Uncertainty", x = "Angle", y = "1 - P(correct)") +
  theme_minimal()

plot_lr <- p_lr_err + p_lr_unc + plot_annotation(title = "Logistic Regression Performance")
print(plot_lr)

df_dt_bin <- bin_summary %>% filter(Model == "Decision_Tree_NestedCV") 

p_dt_err <- ggplot(df_dt_bin, aes(x = angle_mid, y = mean_error)) +
  geom_col(fill = "lightskyblue", color = "black") +
  geom_errorbar(aes(ymin = mean_error - 1.96 * se_error, ymax = mean_error + 1.96 * se_error), width = 2, color = "black", size = 0.8) +
  labs(title = "Actual Error", x = "Angle", y = "Mean Error") +
  theme_minimal()

p_dt_unc <- ggplot(df_dt_bin, aes(x = angle_mid, y = mean_uncertainty)) +
  geom_col(fill = "tomato", color = "black") +
  geom_errorbar(aes(ymin = mean_uncertainty - 1.96 * se_uncertainty, ymax = mean_uncertainty + 1.96 * se_uncertainty), width = 2, color = "black", size = 0.8) +
  labs(title = "Uncertainty", x = "Angle", y = "1 - P(correct)") +
  theme_minimal()

plot_dt <- p_dt_err + p_dt_unc + plot_annotation(title = "Decision Tree Performance")
print(plot_dt)

df_lr <- df %>% filter(Model == "Logistic_Regression")
df_dt <- df %>% filter(Model == "Decision_Tree_NestedCV")

# chisquared test lr
print(chisq.test(table(df_lr$angle_bin, df_lr$error)))

# chisquared test dt
print(chisq.test(table(df_dt$angle_bin, df_dt$error)))

#2b

true_freq <- c("NBA" = 0.6, "EURO" = 0.1, "U16" = 0.1, "U14" = 0.1, "SLO1"=0.1)
emp_freq <- prop.table(table(df$Competition))
print(emp_freq)
print(true_freq)
comp_name <- names(true_freq)
print(comp_name)

weights <- true_freq/as.numeric(emp_freq[comp_name])
print(weights)
df <- df %>% mutate(iw = weights[Competition])

weighted_metrics <- function(sub){
  w <- sub$iw
  y <- sub$y_true
  yh <- sub$y_pred
  p <- sub$prob_correct_class
  acc <- weighted.mean(yh == y, w)
  p_c <- pmax(pmin(p, 1- 1e-7), 1e-7)
  logloss <- - weighted.mean(log(p_c), w)
  tibble(weighted_acc = acc,
         weighted_logloss = logloss)
}
results_weighted <- df %>%
  group_by(Model) %>%
  group_modify(~ weighted_metrics(.x)) %>%
  ungroup()

print(results_weighted)

results_unweighted <- df %>%
  group_by(Model) %>%
  summarise(
    unweighted_accuracy = mean(y_pred == y_true),
    unweighted_logloss  = -mean(log(pmax(pmin(prob_correct_class, 1-1e-7), 1e-7))),
    .groups = "drop"
  )
print(results_unweighted)
comparison <- left_join(results_unweighted, results_weighted, by = "Model")
print(comparison)

