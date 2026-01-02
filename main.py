"""
Automates:
1. validation
2. categorization
3. summarization
of invoice documents using Python (Pandas, OS, OpenPyXL, Matplotlib)
"""
import os
import re
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ===================================
# Configuration 
# ===================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FOLDER = os.path.join(BASE_DIR, "raw_invoices")
VALID_FOLDER = os.path.join(BASE_DIR, "processed", "valid")
INVALID_FOLDER = os.path.join(BASE_DIR, "processed", "invalid")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summary_reports")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")

plots_folder = os.path.join(SUMMARY_FOLDER, "plots")

os.makedirs(VALID_FOLDER, exist_ok=True)
os.makedirs(INVALID_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

ACTIVITY_LOG = os.path.join(LOG_FOLDER, "activity_log.txt")
ERROR_LOG = os.path.join(LOG_FOLDER, "error_log.txt")

# ===================================
# Validation functions 
# ===================================
def validate_email(email):
    # Check if email format is valid
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, str(email)))

def validate_row(row):
    # Return True if a row passes all validation checks
    try:
        valid_amount = float(row['amount']) >= 0
        valid_qty = int(row['qty']) > 0
        valid_email = validate_email(row['email'])
        valid_date = pd.to_datetime(row['invoice_date'], errors='coerce', dayfirst=True) is not pd.NaT
        return all([valid_amount,valid_qty, valid_email, valid_date])
    except Exception:   
        return False

def log_activity(message):
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open(ACTIVITY_LOG, "a") as f:
        f.write(f"[{timestamp}] INFO: {message}\n")

def log_error(message):
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open(ERROR_LOG, "a") as f:
        f.write(f"[{timestamp}] ERROR: {message}\n")

# ===================================
# Helper function 
# ===================================
# 1. Get month for folder naming
def get_month_folder(date_str):
    # Return folder name as 'MMMyyyy' (e.g. Jan2025)
    try:
        date = pd.to_datetime(date_str, dayfirst=True)
        return date.strftime("%b%Y")
    except Exception:
        return "UnknownMonth"
    
# 2. Read and combine all CSV files from the raw folder
def read_invoices(folder_path): 
    combined_df = pd.DataFrame()
    # Find all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError("No invoice CSV files found in the folder.")
    
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print(f"Reading: {file}")
        try:
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            log_activity(f"Successfully loaded in {len(df)} records from {file}")
        except Exception as e:
            log_error(f"Error reading {file}: {e}")
    print(f"\nTotal invoices loaded: {len(combined_df)}")
    return combined_df

# 3. Save valid invoices into monthly folders
def save_valid_invoices(df):
    df["month_folder"] = df["invoice_date"].apply(get_month_folder)
    for month, group in df.groupby("month_folder"):
        month_path = os.path.join(VALID_FOLDER, month)
        os.makedirs(month_path, exist_ok=True)
        output_path = os.path.join(month_path, f"{month}_invoices.csv")
        group.drop(columns=["is_valid", "month_folder"]).to_csv(output_path, index=False)
    
    log_activity(f"Saved valid invoices in {VALID_FOLDER}")

# 4. Save invalid invoices into the invalid folder
def save_invalid_invoices(df):
    output_path = os.path.join(INVALID_FOLDER, "invalid_invoices.csv")
    df.to_csv(output_path, index=False)
    log_activity(f"Saved invalid invoices in {INVALID_FOLDER}")

# 5. Generate financial and validation summary reports
def generate_summary(valid_df, all_df):
    valid_df["total_amount"] = valid_df["qty"].astype(float) * valid_df["amount"].astype(float)

    # Top 20 Invoice Count
    city_counts = valid_df["city"].value_counts()
    top_20_invoice_count = city_counts.head(20).sum()

    # Top 20 Revenue
    revenue_by_city = valid_df.groupby("city")["total_amount"].sum().sort_values(ascending=False)
    top_20_revenue = revenue_by_city.head(20).sum()

    summary = {
        "Total Invoices": [len(all_df)], 
        "Valid Invoices": [len(valid_df)],
        "Invalid Invoices": [len(all_df) - len(valid_df)], 
        "Validation Accuracy (%)": [len(valid_df) / len(all_df) * 100],
        "Total Revenue": [valid_df["total_amount"].sum()], 
        "Average Amount": [valid_df["amount"].mean()], 
        "Top Product ID": [valid_df["product_id"].mode()[0] if not valid_df["product_id"].empty else "N/A"],
        "Top City": [valid_df["city"].mode()[0] if not valid_df["city"].empty else "N/A"],
        "Total Unique Cities": [valid_df["city"].nunique()],
        "Total Invoices (Top 20 Cities)": [top_20_invoice_count],
        "Total Revenue (Top 20 Cities)": [top_20_revenue],
        "Earliest Invoice Date": [valid_df["invoice_date"].min()],
        "Latest Invoice Date": [valid_df["invoice_date"].max()]
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(SUMMARY_FOLDER, "summary_report.csv"), index=False)
    summary_df.to_excel(os.path.join(SUMMARY_FOLDER, "summary_report.xlsx"), index=False)
    log_activity(f"Generated summary_report.csv and summary_report.xlsx")
    return summary_df

# 6. Monthly revenue trend
def generate_monthly_revenue_trend(valid_df):
    plt.figure(figsize=(18,10))
    valid_df["invoice_date"] = pd.to_datetime(valid_df["invoice_date"], errors="coerce")
    monthly_revenue = valid_df.groupby(valid_df["invoice_date"].dt.to_period("M"))["total_amount"].sum()
    monthly_revenue.plot(kind="line", title="Monthly Revenue Trend", linewidth=1.0)
    plt.xlabel("Invoice Date (year)")
    plt.ylabel("Revenue (RM)")
    plt.tight_layout()
    plot_name = "monthly_revenue_trend.png"
    plt.savefig(os.path.join(plots_folder, plot_name),dpi=600)
    plt.close()
    log_activity(f"Saved plot: {plot_name}")

# 7. Save per-city invoice data
def save_citywise_data(valid_df):
    city_folder = os.path.join(SUMMARY_FOLDER, "city_reports")
    os.makedirs(city_folder, exist_ok=True)

    city_summary = valid_df["city"].value_counts().reset_index()
    city_summary.columns = ["City", "Invoice_Count"]

    all_cities_csv = "all_cities_invoice_counts.csv"
    city_summary.to_csv(os.path.join(SUMMARY_FOLDER, all_cities_csv), index=False)
    log_activity(f"Saved complete city list: {all_cities_csv}")

    # Calculate total cities and total invoice count for top 20
    total_unique_cities = len(city_summary)

    top_20_cities = city_summary.head(20)

    top_20_total_invoices = top_20_cities["Invoice_Count"].sum()

    print(f"Total Unique Cities: {total_unique_cities}")
    print(f"Total Invoices Count (Top 20 Cities): {top_20_total_invoices}")
    log_activity(f"Total Unique Cities: {total_unique_cities}")
    log_activity(f"Total Invoices in Top 20 Cities: {top_20_total_invoices}")

    csv_name = "top_20_city_summary.csv"
    top_20_cities.to_csv(os.path.join(SUMMARY_FOLDER, csv_name), index=False)

    # Save each city's invoice list
    for city in top_20_cities["City"]:
        group = valid_df[valid_df["city"] == city]
        city_name = re.sub(r'[\\/*?:"<>|]', "_", city)
        group.to_csv(os.path.join(city_folder, f"{city_name}_invoices.csv"), index=False)

    log_activity(f"Saved {csv_name}")

    # Plot invoice count per city
    plt.figure(figsize=(12,8))
    bars = plt.bar(top_20_cities["City"], top_20_cities["Invoice_Count"], color="#1f77b4")
    plt.bar_label(bars, fmt="%d")
    plt.title("Top 20 Cities by Invoice Count")
    plt.xlabel("City")
    plt.ylabel("Invoice Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plot_name = "top_20_cities_by_invoice_count.png"
    plt.savefig(os.path.join(plots_folder, plot_name), dpi=600)
    plt.close()
    log_activity(f"Saved plot: {plot_name}")

# 8. Revenue per city
def generate_revenue_by_city(valid_df):
    valid_df["total_amount"] = valid_df["qty"].astype(float) * valid_df["amount"].astype(float)
    revenue_by_city = valid_df.groupby("city")["total_amount"].sum().sort_values(ascending=False)

    # Calculate total revenue among top 20 cities
    top_20_revenue_sum = revenue_by_city.head(20).sum()
    print(f"Total Revenue (Top 20 Cities): RM{top_20_revenue_sum:,.2f}")
    log_activity(f"Total Revenue (Top 20 Cities): RM{top_20_revenue_sum:,.2f}")

    plt.figure(figsize=(12,10))
    ax = revenue_by_city.head(20).plot(kind="bar", color="#1f77b4", title="Top 20 Cities by Revenue")
    for container in ax.containers:
        labels = ax.bar_label(container, labels=[f"RM{x.get_height():.2f}" for x in container], padding=3)
        for label in labels:
            label.set_rotation(90)
            label.set_ha("right")
    plt.ylim(0, revenue_by_city.head(20).max() * 1.15)
    plt.xlabel("City")
    plt.ylabel("Total Revenue (RM)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plot_name = "top_20_cities_by_revenue.png"
    plt.savefig(os.path.join(plots_folder, plot_name), dpi=600)
    plt.close()
    log_activity(f"Saved plot: {plot_name}")

# 9. Efficency comparison summary (as a dataframe)
def save_efficiency_report(manual_time, automated_time, total_invoices):
    efficiency = ((manual_time - automated_time) / manual_time) * 100
    df = pd.DataFrame({
        "Metric": ["Manual Time (s)", "Automated Time (s)", "Time Saved (s)", "Efficiency (%)"],
        "Value": [manual_time, automated_time, manual_time - automated_time, efficiency]
    })
    df.to_csv(os.path.join(SUMMARY_FOLDER, "efficiency_report.csv"), index=False)
    return efficiency

# 10. Create a bar chart comparing manual vs. automated time
def plot_efficiency_chart(manual_time, automated_time):
    plt.figure(figsize=(10,8))
    bars = plt.bar(["Manual", "Automated"], [manual_time, automated_time], color=["#e74c3c", "#1f77b4"])
    plt.yscale("log")
    plt.ylabel("Processing Time (seconds, log scale)")
    plt.title("Invoice Processing Efficiency (Log Scale)")
    plt.bar_label(bars, labels=[f"{manual_time:.2f}s", f"{automated_time:.2f}s"], padding=5)
    plt.tight_layout()
    plot_name = "efficiency_log_scale.png"
    plt.savefig(os.path.join(plots_folder, plot_name), dpi=600)
    plt.close()
    log_activity(f"Saved plot: {plot_name}")

# 11. Mean revenue by year
def generate_mean_revenue_by_decade(valid_df):
    valid_df["invoice_date"] = pd.to_datetime(valid_df["invoice_date"], errors="coerce")
    valid_df["year"] = valid_df["invoice_date"].dt.year
    valid_df["total_amount"] = valid_df["qty"].astype(float) * valid_df["amount"].astype(float)

    valid_df = valid_df.dropna(subset=["year"])

    # Compute total revenue per year
    revenue_per_year = (
        valid_df.groupby("year")["total_amount"]
        .sum()
        .sort_index()
    )

    # Group years into 10-year intervals (decades)
    decade_groups = (revenue_per_year.index // 10) * 10
    revenue_per_year.index = decade_groups

    # Sum revenue for each decade and divide by 10 to get average per year
    mean_revenue_by_decade = revenue_per_year.groupby(revenue_per_year.index).sum() / 10

    # Create readable labels (e.g., "2000-2009")
    decade_labels = [f"{int(y)} - {int(y+9)}" for y in mean_revenue_by_decade.index]

    plt.figure(figsize=(10,8))
    bars = plt.bar(decade_labels, mean_revenue_by_decade.values, color="#1f77b4")
    plt.bar_label(bars, labels=[f"RM{v:,.2f}" for v in mean_revenue_by_decade.values], padding=3)
    plt.title("Mean Revenue by 10-Year Interval")
    plt.xlabel("Decade")
    plt.ylabel("Mean Revenue (RM)")
    plt.tight_layout()

    plot_name = "mean_revenue_by_decade.png"
    plt.savefig(os.path.join(plots_folder, plot_name), dpi=600)
    plt.close()

    log_activity(f"Saved plot: {plot_name}")

# ===================================
# Main Processing Logic 
# ===================================
def process_invoices():
    print("Starting automated invoice processing...\n")
    start_time = time.time()

    # Read data
    all_invoices = read_invoices(RAW_FOLDER)
    print(f"Total invoices read: {len(all_invoices)}")

    # Validate data
    all_invoices["is_valid"] = all_invoices.apply(validate_row, axis=1)
    valid_df = all_invoices[all_invoices["is_valid"]]
    invalid_df = all_invoices[~all_invoices["is_valid"]]

    print(f"Valid invoices: {len(valid_df)}")
    print(f"Invalid invoices: {len(invalid_df)}")

    # Save categorized files
    save_valid_invoices(valid_df)
    save_invalid_invoices(invalid_df)

    # Generate reports
    summary_df = generate_summary(valid_df, all_invoices)

    # More visuals
    generate_monthly_revenue_trend(valid_df)
    save_citywise_data(valid_df)
    generate_revenue_by_city(valid_df)
    generate_mean_revenue_by_decade(valid_df)

    # Efficiency measurement
    manual_time = len(all_invoices) * 1200 # seconds per invoice manually
    automated_time = time.time() - start_time
    efficiency = save_efficiency_report(manual_time, automated_time, len(all_invoices))
    plot_efficiency_chart(manual_time, automated_time)

    # Combine all in one dashboard
    dashboard_path = os.path.join(SUMMARY_FOLDER, "dashboard.xlsx")
    with pd.ExcelWriter(dashboard_path) as writer:
        valid_df.to_excel(writer, sheet_name="Valid Invoices", index=False)
        invalid_df.to_excel(writer, sheet_name="Invalid Invoices", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    # Final summary
    elapsed = time.time() - start_time
    print("\n================= PROCESSING SUMMARY =================")
    print(f"Total invoices processed: {len(all_invoices)}")
    print(f"Valid invoices: {len(valid_df)}")
    print(f"Invalid invoices: {len(invalid_df)}")
    print(f"Total revenue: RM{valid_df['qty'].astype(float).mul(valid_df['amount'].astype(float)).sum():,.2f}")
    print(f"Efficiency improvement: {efficiency:.4f}% faster")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print(f"Reports and charts saved in: {SUMMARY_FOLDER}")
    print("======================================================\n")

    log_activity("All processing completed successfully.")
    log_activity(f"Reports and charts saved in {SUMMARY_FOLDER}")

if __name__ == "__main__":
    try:
        process_invoices()
    except Exception as e:
        log_error(f"Fatal error: {e}")
        print(f"An error occurred. Check {ERROR_LOG} for details.")
