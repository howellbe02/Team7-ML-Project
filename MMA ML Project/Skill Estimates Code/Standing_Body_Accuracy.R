library(readxl)
library(writexl)
library(arm)
library(igraph)

df <- read_excel("C:/Users/danie/Desktop/Test MMA Project/MMA ML Project/data/NewMMAMatches.xlsx")
df$Date <- as.Date(df$Date, format = "%B %d, %Y")
df <- df[df$Date < cutoff_date, ]
unique_weights <- setdiff(unique(df$`Cleaned Weight Class`), c("Catch Weight", "Unknown"))

for (weight in unique_weights) {
  cat("Processing:", weight, "\n")
  
  data_subset <- df[df$`Cleaned Weight Class` == weight, ]
  data_subset$`Fighter i ID` <- as.character(data_subset$`Fighter i ID`)
  data_subset$`Fighter j ID` <- as.character(data_subset$`Fighter j ID`)
  
  g <- graph_from_data_frame(data_subset[, c("Fighter i ID", "Fighter j ID")], directed = FALSE)
  comp <- components(g)
  largest_membership <- which.max(comp$csize)
  core_fighters <- as.character(V(g)$name[comp$membership == largest_membership])
  
  data_subset <- data_subset[
    data_subset$`Fighter i ID` %in% core_fighters &
      data_subset$`Fighter j ID` %in% core_fighters, ]
  
  data_subset <- data_subset[
    data_subset$`Match Length Sec` >= 60 &
      !is.na(data_subset$`Standing Body Shots Attempted`) &
      !is.na(data_subset$`Standing Body Shots Landed`) &
      data_subset$`Standing Body Shots Attempted` > 0 &
      data_subset$`Standing Body Shots Landed` <= data_subset$`Standing Body Shots Attempted`, ]
  
  if (nrow(data_subset) < 10) {
    cat("  Skipping", weight, "- too few valid fights\n")
    next
  }
  
  model <- bayesglm(
    cbind(data_subset$`Standing Body Shots Landed`,
          data_subset$`Standing Body Shots Attempted` - data_subset$`Standing Body Shots Landed`) ~ 
      factor(data_subset$`Fighter i ID`) + factor(data_subset$`Fighter j ID`),
    family = binomial(link = "logit")
  )
  
  coefs <- summary(model)$coefficients
  offense_idx <- grepl("Fighter i ID", rownames(coefs))
  defense_idx <- grepl("Fighter j ID", rownames(coefs))
  
  offense_df <- data.frame(
    Fighter_ID = gsub(".*`Fighter i ID`\\)", "", rownames(coefs)[offense_idx]),
    coefs[offense_idx, ]
  )
  
  defense_df <- data.frame(
    Fighter_ID = gsub(".*`Fighter j ID`\\)", "", rownames(coefs)[defense_idx]),
    coefs[defense_idx, ]
  )
  
  folder <- paste0(output_root, weight, "/")
  write_xlsx(offense_df, paste0(folder, "Offense_Standing_Body_Accuracy.xlsx"))
  write_xlsx(defense_df, paste0(folder, "Defense_Standing_Body_Accuracy.xlsx"))
}