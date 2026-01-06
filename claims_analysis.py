import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("        INSURANCE CLAIMS ANALYSIS - AUTO INSURANCE DATASET")
print("="*70)

# ============================================
# 1. LOAD AND EXPLORE DATA
# ============================================
print("\n1. Loading data...")
df = pd.read_csv('insurance_claims.csv')

print(f"\n✓ Data loaded successfully!")
print(f"  - Total claims: {len(df):,}")
print(f"  - Features: {df.shape[1]}")

print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)
print(df.head())

print("\n" + "="*70)
print("COLUMN NAMES")
print("="*70)
print(df.columns.tolist())

print("\n" + "="*70)
print("DATA TYPES")
print("="*70)
print(df.dtypes)

print("\n" + "="*70)
print("MISSING VALUES")
print("="*70)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing[missing > 0])

# ============================================
# 2. BASIC STATISTICS
# ============================================
print("\n" + "="*70)
print("BASIC STATISTICS - CLAIM AMOUNTS")
print("="*70)
print(df[['total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']].describe())

# ============================================
# 3. CLAIMS ANALYSIS BY INCIDENT TYPE
# ============================================
print("\n" + "="*70)
print("ANALYSIS 1: CLAIMS BY INCIDENT TYPE")
print("="*70)

claims_by_type = df.groupby('incident_type').agg({
    'total_claim_amount': ['count', 'mean', 'sum', 'median', 'max']
}).round(2)
claims_by_type.columns = ['Count', 'Avg_Amount', 'Total_Amount', 'Median_Amount', 'Max_Amount']
claims_by_type = claims_by_type.sort_values('Total_Amount', ascending=False)
print(claims_by_type)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Count by incident type
incident_counts = df['incident_type'].value_counts()
axes[0, 0].bar(incident_counts.index, incident_counts.values, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Number of Claims by Incident Type', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Incident Type')
axes[0, 0].set_ylabel('Number of Claims')
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(incident_counts.values):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Plot 2: Average amount by incident type
avg_by_type = df.groupby('incident_type')['total_claim_amount'].mean().sort_values(ascending=False)
axes[0, 1].bar(avg_by_type.index, avg_by_type.values, color='coral', edgecolor='black')
axes[0, 1].set_title('Average Claim Amount by Incident Type', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Incident Type')
axes[0, 1].set_ylabel('Average Amount ($)')
axes[0, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(avg_by_type.values):
    axes[0, 1].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

# Plot 3: Total claim value by incident type
total_by_type = df.groupby('incident_type')['total_claim_amount'].sum().sort_values(ascending=False)
axes[1, 0].bar(total_by_type.index, total_by_type.values, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Total Claim Value by Incident Type', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Incident Type')
axes[1, 0].set_ylabel('Total Amount ($)')
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(total_by_type.values):
    axes[1, 0].text(i, v + 50000, f'${v/1000:,.0f}K', ha='center', fontweight='bold')

# Plot 4: Incident severity distribution
severity_counts = df['incident_severity'].value_counts()
axes[1, 1].bar(severity_counts.index, severity_counts.values, color='lightcoral', edgecolor='black')
axes[1, 1].set_title('Claims by Incident Severity', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Severity')
axes[1, 1].set_ylabel('Number of Claims')
for i, v in enumerate(severity_counts.values):
    axes[1, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('claims_by_incident_type.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: claims_by_incident_type.png")
plt.close()

# ============================================
# 4. CLAIM AMOUNT DISTRIBUTION
# ============================================
print("\n" + "="*70)
print("ANALYSIS 2: CLAIM AMOUNT DISTRIBUTION")
print("="*70)

print(f"Average total claim: ${df['total_claim_amount'].mean():,.2f}")
print(f"Median total claim: ${df['total_claim_amount'].median():,.2f}")
print(f"Total claims value: ${df['total_claim_amount'].sum():,.2f}")
print(f"Max claim: ${df['total_claim_amount'].max():,.2f}")
print(f"Min claim: ${df['total_claim_amount'].min():,.2f}")
print(f"\nClaim breakdown:")
print(f"  - Average injury claim: ${df['injury_claim'].mean():,.2f}")
print(f"  - Average property claim: ${df['property_claim'].mean():,.2f}")
print(f"  - Average vehicle claim: ${df['vehicle_claim'].mean():,.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram of total claims
axes[0, 0].hist(df['total_claim_amount'], bins=40, color='lightgreen', edgecolor='black')
axes[0, 0].set_title('Distribution of Total Claim Amounts', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Total Claim Amount ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['total_claim_amount'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["total_claim_amount"].mean():,.0f}')
axes[0, 0].axvline(df['total_claim_amount'].median(), color='blue', linestyle='--', linewidth=2, label=f'Median: ${df["total_claim_amount"].median():,.0f}')
axes[0, 0].legend()

# Box plot
axes[0, 1].boxplot(df['total_claim_amount'])
axes[0, 1].set_title('Total Claim Amount Box Plot', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Total Claim Amount ($)')

# Claim components breakdown
claim_components = ['injury_claim', 'property_claim', 'vehicle_claim']
avg_components = [df[col].mean() for col in claim_components]
axes[1, 0].bar(['Injury', 'Property', 'Vehicle'], avg_components, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
axes[1, 0].set_title('Average Claim by Component Type', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Average Amount ($)')
for i, v in enumerate(avg_components):
    axes[1, 0].text(i, v + 200, f'${v:,.0f}', ha='center', fontweight='bold')

# Scatter: Age vs Total Claim
axes[1, 1].scatter(df['age'], df['total_claim_amount'], alpha=0.5, color='purple')
axes[1, 1].set_title('Age vs Total Claim Amount', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Customer Age')
axes[1, 1].set_ylabel('Total Claim Amount ($)')

plt.tight_layout()
plt.savefig('claim_amount_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: claim_amount_analysis.png")
plt.close()

# ============================================
# 5. FRAUD ANALYSIS
# ============================================
print("\n" + "="*70)
print("ANALYSIS 3: FRAUD DETECTION ANALYSIS")
print("="*70)

fraud_stats = df.groupby('fraud_reported').agg({
    'total_claim_amount': ['count', 'mean', 'sum', 'max']
}).round(2)
fraud_stats.columns = ['Count', 'Avg_Amount', 'Total_Amount', 'Max_Amount']
print(fraud_stats)

fraud_rate = (df['fraud_reported'] == 'Y').sum() / len(df) * 100
print(f"\n⚠️  FRAUD RATE: {fraud_rate:.2f}%")
print(f"   Fraudulent claims: {(df['fraud_reported'] == 'Y').sum()}")
print(f"   Legitimate claims: {(df['fraud_reported'] == 'N').sum()}")

# Fraud by incident type
print("\nFraud distribution by incident type:")
fraud_by_incident = pd.crosstab(df['incident_type'], df['fraud_reported'], normalize='index') * 100
print(fraud_by_incident.round(2))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Pie chart: Fraud vs Non-Fraud
fraud_counts = df['fraud_reported'].value_counts()
colors_fraud = ['#90EE90', '#FF6B6B']
axes[0, 0].pie(fraud_counts.values, labels=['Legitimate', 'Fraudulent'], autopct='%1.1f%%', 
               colors=colors_fraud, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0, 0].set_title('Fraud vs Non-Fraud Claims', fontsize=14, fontweight='bold')

# Bar chart: Average claim amount (fraud vs non-fraud)
fraud_avg = df.groupby('fraud_reported')['total_claim_amount'].mean()
axes[0, 1].bar(['Legitimate', 'Fraudulent'], fraud_avg.values, color=colors_fraud, edgecolor='black')
axes[0, 1].set_title('Average Claim: Fraud vs Legitimate', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Average Claim Amount ($)')
for i, v in enumerate(fraud_avg.values):
    axes[0, 1].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

# Fraud by incident type
fraud_by_type = df[df['fraud_reported'] == 'Y']['incident_type'].value_counts()
axes[1, 0].bar(fraud_by_type.index, fraud_by_type.values, color='#FF6B6B', edgecolor='black')
axes[1, 0].set_title('Fraudulent Claims by Incident Type', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Incident Type')
axes[1, 0].set_ylabel('Number of Fraudulent Claims')
axes[1, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(fraud_by_type.values):
    axes[1, 0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# Fraud by severity
fraud_by_severity = df[df['fraud_reported'] == 'Y']['incident_severity'].value_counts()
axes[1, 1].bar(fraud_by_severity.index, fraud_by_severity.values, color='#FFB6B9', edgecolor='black')
axes[1, 1].set_title('Fraudulent Claims by Severity', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Incident Severity')
axes[1, 1].set_ylabel('Number of Fraudulent Claims')
for i, v in enumerate(fraud_by_severity.values):
    axes[1, 1].text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('fraud_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: fraud_analysis.png")
plt.close()

# ============================================
# 6. DEMOGRAPHIC ANALYSIS
# ============================================
print("\n" + "="*70)
print("ANALYSIS 4: DEMOGRAPHIC PATTERNS")
print("="*70)

# Age analysis
df['age_group'] = pd.cut(df['age'], 
                          bins=[0, 25, 35, 45, 55, 65, 100],
                          labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])

age_analysis = df.groupby('age_group').agg({
    'total_claim_amount': ['count', 'mean', 'sum']
}).round(2)
age_analysis.columns = ['Count', 'Avg_Amount', 'Total_Amount']
print("\nClaims by Age Group:")
print(age_analysis)

# Gender analysis
print("\nClaims by Gender:")
gender_analysis = df.groupby('insured_sex').agg({
    'total_claim_amount': ['count', 'mean']
}).round(2)
gender_analysis.columns = ['Count', 'Avg_Amount']
print(gender_analysis)

# Education level
print("\nClaims by Education Level:")
education_analysis = df.groupby('insured_education_level').agg({
    'total_claim_amount': ['count', 'mean']
}).round(2)
education_analysis.columns = ['Count', 'Avg_Amount']
print(education_analysis)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Age group analysis
age_avg = df.groupby('age_group')['total_claim_amount'].mean()
axes[0, 0].bar(range(len(age_avg)), age_avg.values, color='mediumpurple', edgecolor='black')
axes[0, 0].set_xticks(range(len(age_avg)))
axes[0, 0].set_xticklabels(age_avg.index, rotation=45)
axes[0, 0].set_title('Average Claim by Age Group', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Average Claim Amount ($)')
for i, v in enumerate(age_avg.values):
    axes[0, 0].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

# Gender analysis
gender_avg = df.groupby('insured_sex')['total_claim_amount'].mean()
axes[0, 1].bar(['Female', 'Male'], gender_avg.values, color=['#FF69B4', '#4169E1'], edgecolor='black')
axes[0, 1].set_title('Average Claim by Gender', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Average Claim Amount ($)')
for i, v in enumerate(gender_avg.values):
    axes[0, 1].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

# Education level
edu_avg = df.groupby('insured_education_level')['total_claim_amount'].mean().sort_values(ascending=False)
axes[1, 0].barh(range(len(edu_avg)), edu_avg.values, color='teal', edgecolor='black')
axes[1, 0].set_yticks(range(len(edu_avg)))
axes[1, 0].set_yticklabels(edu_avg.index)
axes[1, 0].set_title('Average Claim by Education Level', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Average Claim Amount ($)')
for i, v in enumerate(edu_avg.values):
    axes[1, 0].text(v + 500, i, f'${v:,.0f}', va='center', fontweight='bold')

# Customer tenure
axes[1, 1].scatter(df['months_as_customer'], df['total_claim_amount'], alpha=0.5, color='orange')
axes[1, 1].set_title('Customer Tenure vs Claim Amount', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Months as Customer')
axes[1, 1].set_ylabel('Total Claim Amount ($)')

plt.tight_layout()
plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: demographic_analysis.png")
plt.close()

# ============================================
# 7. COLLISION & VEHICLE ANALYSIS
# ============================================
print("\n" + "="*70)
print("ANALYSIS 5: COLLISION & VEHICLE PATTERNS")
print("="*70)

# Collision type analysis
print("\nClaims by Collision Type:")
collision_analysis = df.groupby('collision_type').agg({
    'total_claim_amount': ['count', 'mean']
}).round(2)
collision_analysis.columns = ['Count', 'Avg_Amount']
print(collision_analysis)

# Vehicle make analysis
print("\nTop 10 Vehicle Makes by Claim Frequency:")
top_makes = df['auto_make'].value_counts().head(10)
print(top_makes)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Collision type
collision_avg = df.groupby('collision_type')['total_claim_amount'].mean().sort_values(ascending=False)
axes[0, 0].bar(range(len(collision_avg)), collision_avg.values, color='crimson', edgecolor='black')
axes[0, 0].set_xticks(range(len(collision_avg)))
axes[0, 0].set_xticklabels(collision_avg.index, rotation=45, ha='right')
axes[0, 0].set_title('Average Claim by Collision Type', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Average Claim Amount ($)')
for i, v in enumerate(collision_avg.values):
    axes[0, 0].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

# Number of vehicles involved
vehicles_avg = df.groupby('number_of_vehicles_involved')['total_claim_amount'].mean()
axes[0, 1].bar(vehicles_avg.index.astype(str), vehicles_avg.values, color='darkorange', edgecolor='black')
axes[0, 1].set_title('Average Claim by Vehicles Involved', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Vehicles')
axes[0, 1].set_ylabel('Average Claim Amount ($)')
for i, v in enumerate(vehicles_avg.values):
    axes[0, 1].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

# Top vehicle makes
top_makes_10 = df['auto_make'].value_counts().head(10)
axes[1, 0].barh(range(len(top_makes_10)), top_makes_10.values, color='steelblue', edgecolor='black')
axes[1, 0].set_yticks(range(len(top_makes_10)))
axes[1, 0].set_yticklabels(top_makes_10.index)
axes[1, 0].set_title('Top 10 Vehicle Makes by Claim Count', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Number of Claims')
for i, v in enumerate(top_makes_10.values):
    axes[1, 0].text(v + 1, i, str(v), va='center', fontweight='bold')

# Property damage
property_damage_avg = df.groupby('property_damage')['total_claim_amount'].mean()
axes[1, 1].bar(['No Damage', 'Damage'], property_damage_avg.values, color=['lightgreen', 'salmon'], edgecolor='black')
axes[1, 1].set_title('Average Claim: Property Damage vs No Damage', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Average Claim Amount ($)')
for i, v in enumerate(property_damage_avg.values):
    axes[1, 1].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('collision_vehicle_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: collision_vehicle_analysis.png")
plt.close()

# ============================================
# 8. TIME & LOCATION ANALYSIS
# ============================================
print("\n" + "="*70)
print("ANALYSIS 6: TIME & LOCATION PATTERNS")
print("="*70)

# Hour of day analysis
print("\nClaims by Hour of Day (Top 5):")
hour_analysis = df.groupby('incident_hour_of_the_day').agg({
    'total_claim_amount': ['count', 'mean']
}).round(2)
hour_analysis.columns = ['Count', 'Avg_Amount']
print(hour_analysis.sort_values('Count', ascending=False).head())

# State analysis
print("\nTop 10 States by Claim Count:")
state_counts = df['incident_state'].value_counts().head(10)
print(state_counts)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Claims by hour
hour_counts = df['incident_hour_of_the_day'].value_counts().sort_index()
axes[0, 0].plot(hour_counts.index, hour_counts.values, marker='o', color='darkblue', linewidth=2, markersize=6)
axes[0, 0].set_title('Claims by Hour of Day', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Hour of Day (0-23)')
axes[0, 0].set_ylabel('Number of Claims')
axes[0, 0].grid(True, alpha=0.3)

# Top states
top_states = df['incident_state'].value_counts().head(10)
axes[0, 1].barh(range(len(top_states)), top_states.values, color='forestgreen', edgecolor='black')
axes[0, 1].set_yticks(range(len(top_states)))
axes[0, 1].set_yticklabels(top_states.index)
axes[0, 1].set_title('Top 10 States by Claim Count', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Claims')
for i, v in enumerate(top_states.values):
    axes[0, 1].text(v + 1, i, str(v), va='center', fontweight='bold')

# Police report availability
police_avg = df.groupby('police_report_available')['total_claim_amount'].mean()
axes[1, 0].bar(['Not Available', 'Available'], police_avg.values, color=['#FFB347', '#77DD77'], edgecolor='black')
axes[1, 0].set_title('Average Claim: Police Report Availability', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Average Claim Amount ($)')
for i, v in enumerate(police_avg.values):
    axes[1, 0].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

# Authorities contacted
authorities_counts = df['authorities_contacted'].value_counts()
axes[1, 1].bar(authorities_counts.index, authorities_counts.values, color='indianred', edgecolor='black')
axes[1, 1].set_title('Claims by Authorities Contacted', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Authority Type')
axes[1, 1].set_ylabel('Number of Claims')
axes[1, 1].tick_params(axis='x', rotation=45)
for i, v in enumerate(authorities_counts.values):
    axes[1, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('time_location_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: time_location_analysis.png")
plt.close()

# ============================================
# 9. KEY BUSINESS INSIGHTS SUMMARY
# ============================================
print("\n" + "="*70)
print("        KEY BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*70)

print("\n📊 OVERALL STATISTICS:")
print(f"  • Total Claims Analyzed: {len(df):,}")
print(f"  • Total Claim Value: ${df['total_claim_amount'].sum():,.2f}")
print(f"  • Average Claim: ${df['total_claim_amount'].mean():,.2f}")
print(f"  • Median Claim: ${df['total_claim_amount'].median():,.2f}")

print("\n🚗 INCIDENT PATTERNS:")
most_common_incident = df['incident_type'].value_counts().idxmax()
print(f"  • Most Common Incident: {most_common_incident} ({df['incident_type'].value_counts().max()} cases)")
highest_avg_incident = df.groupby('incident_type')['total_claim_amount'].mean().idxmax()
highest_avg_amount = df.groupby('incident_type')['total_claim_amount'].mean().max()
print(f"  • Highest Average Claim: {highest_avg_incident} (${highest_avg_amount:,.2f})")
most_severe = df['incident_severity'].value_counts().idxmax()
print(f"  • Most Common Severity: {most_severe}")

print("\n⚠️  FRAUD DETECTION:")
fraud_count = (df['fraud_reported'] == 'Y').sum()
fraud_rate = fraud_count / len(df) * 100
print(f"  • Fraudulent Claims: {fraud_count} ({fraud_rate:.2f}%)")
fraud_avg = df[df['fraud_reported'] == 'Y']['total_claim_amount'].mean()
legit_avg = df[df['fraud_reported'] == 'N']['total_claim_amount'].mean()
print(f"  • Avg Fraudulent Claim: ${fraud_avg:,.2f}")
print(f"  • Avg Legitimate Claim: ${legit_avg:,.2f}")
print(f"  • Difference: ${abs(fraud_avg - legit_avg):,.2f}")

print("\n👥 CUSTOMER DEMOGRAPHICS:")
avg_age = df['age'].mean()
print(f"  • Average Customer Age: {avg_age:.1f} years")
avg_tenure = df['months_as_customer'].mean()
print(f"  • Average Customer Tenure: {avg_tenure:.1f} months")
gender_split = df['insured_sex'].value_counts()
print(f"  • Gender Split: Female ({gender_split.get('FEMALE', 0)}), Male ({gender_split.get('MALE', 0)})")

print("\n🚨 HIGH-RISK INDICATORS:")
print(f"  • Peak Claim Hour: {df['incident_hour_of_the_day'].mode()[0]}:00")
most_collision = df['collision_type'].value_counts().idxmax()
print(f"  • Most Common Collision: {most_collision}")
multi_vehicle_pct = (df['number_of_vehicles_involved'] > 1).sum() / len(df) * 100
print(f"  • Multi-Vehicle Incidents: {multi_vehicle_pct:.1f}%")

print("\n💡 BUSINESS RECOMMENDATIONS:")
print("  1. Focus fraud detection on incident types with highest fraud rates")
print("  2. Adjust premiums based on collision type risk profiles")
print("  3. Implement time-based risk assessment (peak hours)")
print("  4. Target customer education for high-risk age groups")
print("  5. Enhance claims process for multi-vehicle incidents")
print("  6. Consider geographic risk factors in pricing models")

print("\n" + "="*70)
print("✓ ANALYSIS COMPLETE!")
print(f"✓ Generated 5 visualization files:")
print("  - claims_by_incident_type.png")
print("  - claim_amount_analysis.png")
print("  - fraud_analysis.png")
print("  - demographic_analysis.png")
print("  - collision_vehicle_analysis.png")
print("  - time_location_analysis.png")
print("="*70)