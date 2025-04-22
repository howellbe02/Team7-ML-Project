setwd("C:/Users/danie/Desktop/Test MMA Project/MMA ML Project/Skill Estimates Code")
cutoff_date <- as.Date("2023-05-05")
output_root <- paste0("C:/Users/danie/Desktop/Test MMA Project/MMA ML Project/Skill Estimates/Skill Estimates Pre-", cutoff_date, "/")

if (!dir.exists(output_root)) {
  dir.create(output_root)
}

df <- readxl::read_excel("C:/Users/danie/Desktop/Test MMA Project/MMA ML Project/data/NewMMAMatches.xlsx")
unique_weights <- setdiff(unique(df$`Cleaned Weight Class`), c("Catch Weight", "Unknown"))

for (weight in unique_weights) {
  weight_folder <- paste0(output_root, weight, "/")
  if (!dir.exists(weight_folder)) {
    dir.create(weight_folder)
  }
}

source("Standing_Head_Accuracy.R", local = TRUE)
source("Standing_Body_Accuracy.R", local = TRUE)
source("Ground_Head_Accuracy.R", local = TRUE)
source("Ground_Body_Accuracy.R", local = TRUE)
source("Takedown_Accuracy.R", local = TRUE)
source("Submission_Accuracy.R", local = TRUE)
#source("Knockout_Probability.R", local = TRUE)