#!/usr/bin/env python3
"""
Generate realistic Ireland CSO 2011 Census Dataset and populate SQLite/CSV.
Table: pub_combined_view
"""
import pandas as pd
import numpy as np
import sqlite3

def generate_cso_data():
    counties_info = [
        # (County, NUTS III, Planning Region, approx pop 2011)
        ("Dublin", "Dublin", "Eastern and Midland", 1273069),
        ("Cork", "South-West", "Southern", 519032),
        ("Galway", "West", "Northern and Western", 250541),
        ("Kildare", "Mid-East", "Eastern and Midland", 210312),
        ("Limerick", "Mid-West", "Southern", 191809),
        ("Meath", "Mid-East", "Eastern and Midland", 184135),
        ("Donegal", "Border", "Northern and Western", 161137),
        ("Tipperary", "Mid-West", "Southern", 158754),
        ("Kerry", "South-West", "Southern", 145502),
        ("Wexford", "South-East", "Southern", 145320),
        ("Wicklow", "Mid-East", "Eastern and Midland", 136648),
        ("Mayo", "West", "Northern and Western", 130638),
        ("Louth", "Border", "Northern and Western", 122897),
        ("Clare", "Mid-West", "Southern", 117196),
        ("Waterford", "South-East", "Southern", 113795),
        ("Kilkenny", "South-East", "Southern", 95419),
        ("Westmeath", "Midland", "Eastern and Midland", 86164),
        ("Laois", "Midland", "Eastern and Midland", 80559),
        ("Offaly", "Midland", "Eastern and Midland", 76687),
        ("Cavan", "Border", "Northern and Western", 73183),
        ("Sligo", "Border", "Northern and Western", 65393),
        ("Roscommon", "West", "Northern and Western", 64065),
        ("Monaghan", "Border", "Northern and Western", 60483),
        ("Carlow", "South-East", "Southern", 54612),
        ("Longford", "Midland", "Eastern and Midland", 39000),
        ("Leitrim", "Border", "Northern and Western", 31798),
    ]

    records = []
    np.random.seed(42)

    ed_code_counter = 1000

    for county, nuts_iii, planning_region, pop in counties_info:
        # Create multiple Electoral Divisions (EDs) per county for realistic granular distribution
        num_eds = max(3, int(pop / 40000))
        pop_per_ed = pop / num_eds
        for i in range(num_eds):
            ed_pop = int(pop_per_ed * np.random.uniform(0.85, 1.15))
            # Labor force proportions
            working_ratio = np.random.uniform(0.42, 0.48)
            total_working = int(ed_pop * working_ratio)
            total_dependent = ed_pop - total_working
            
            females_working_ratio = np.random.uniform(0.45, 0.49)
            females_working = int(total_working * females_working_ratio)
            males_working = total_working - females_working
            
            females_dependent_ratio = np.random.uniform(0.51, 0.55)
            females_dependent = int(total_dependent * females_dependent_ratio)
            
            # Industry distribution
            agri_ratio = 0.02 if county == "Dublin" else np.random.uniform(0.06, 0.14)
            const_ratio = np.random.uniform(0.05, 0.08)
            mfg_ratio = np.random.uniform(0.11, 0.18)
            comm_ratio = np.random.uniform(0.20, 0.26)
            trans_ratio = np.random.uniform(0.05, 0.09)
            pub_admin_ratio = np.random.uniform(0.06, 0.12)
            
            agri = int(total_working * agri_ratio)
            const = int(total_working * const_ratio)
            mfg = int(total_working * mfg_ratio)
            comm = int(total_working * comm_ratio)
            trans = int(total_working * trans_ratio)
            pub = int(total_working * pub_admin_ratio)
            
            # Education field of study
            study_eng_mfg = int(mfg * np.random.uniform(0.35, 0.55))
            
            ed_code_counter += 1
            records.append({
                "id": len(records) + 1,
                "electoral_division_cso_code": f"ED{ed_code_counter}",
                "county": county,
                "nuts_iii": nuts_iii,
                "planning_region": planning_region,
                "total_working": total_working,
                "total_dependent": total_dependent,
                "males_working": males_working,
                "females_working": females_working,
                "females_dependent": females_dependent,
                "total_agriculture_forestry_fishing": agri,
                "total_building_construction": const,
                "total_manufacturing_industries": mfg,
                "total_commerce_trade": comm,
                "total_transport_communications": trans,
                "total_public_administration": pub,
                "persons_field_of_study_engineering_manufacturing_construction": study_eng_mfg
            })

    df = pd.DataFrame(records)
    
    # Save to CSV
    csv_path = "pub_combined_view.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} records in {csv_path}")

    # Save to SQLite database so sqlalchemy can query it directly
    db_path = "project.db"
    conn = sqlite3.connect(db_path)
    df.to_sql("pub_combined_view", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Stored table 'pub_combined_view' in SQLite database {db_path}")
    return df

if __name__ == "__main__":
    generate_cso_data()
