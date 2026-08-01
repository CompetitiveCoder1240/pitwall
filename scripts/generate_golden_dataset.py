import pickle
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARENTS_PKL = PROJECT_ROOT / "parents.pkl"
OUTPUT_JSON = PROJECT_ROOT / "golden_dataset_50.json"

with open(PARENTS_PKL, "rb") as f:
    vault = pickle.load(f)

# Helper function to find text snippets in vault
def get_text(keyword, section=None):
    matches = []
    for text in vault.values():
        if section and section not in text:
            continue
        if keyword.lower() in text.lower():
            matches.append(text)
    return matches

# Print stats
print(f"Vault contains {len(vault)} parent chunks.")

dataset = [
    # --- SECTION A: GENERAL REGULATORY PROVISIONS ---
    {
        "question": "Which body is responsible for the sporting organisation and regulation of the FIA Formula One World Championship?",
        "ground_truth": "The FIA is responsible for the sporting organisation and regulation of the FIA Formula One World Championship.",
        "expected_article": "Section A, Article A1.1.1"
    },
    {
        "question": "What is the maximum number of Competitions that may take place in a Championship season?",
        "ground_truth": "The maximum number of Competitions permitted in a Championship season is 24.",
        "expected_article": "Section A, Article A3.1.2"
    },
    {
        "question": "What is the deadline for an F1 Team to submit an application and undertake to pay the Entry Fee to the FIA?",
        "ground_truth": "Applications by F1 Teams must be submitted with an undertaking to pay the Entry Fee to the FIA by no later than 15 December of Year N-1.",
        "expected_article": "Section A, Article A3.2.1"
    },
    {
        "question": "What shareholding percentage triggers the Fit and Proper Persons Test for a natural person in an F1 Team or PU Manufacturer?",
        "ground_truth": "The Fit and Proper Persons Test applies to any natural person holding (directly or indirectly) thirty per cent (30%) or more of the outstanding shares of an F1 Team, New Entrant Team, PU Manufacturer, or New Entrant PU Manufacturer.",
        "expected_article": "Section A, Article A4.1.2(a)"
    },
    {
        "question": "What instrument may the Cost Cap Administration enter into with a team or PU Manufacturer to resolve an alleged breach of Financial Regulations without a hearing?",
        "ground_truth": "The Cost Cap Administration may enter into an Accepted Breach Agreement (ABA) to resolve the matter without a hearing.",
        "expected_article": "Section A, Article A7.1.2"
    },
    {
        "question": "When do the 2026 FIA Formula 1 Regulations come into force?",
        "ground_truth": "The FIA F1 Regulations come into force on 1 January 2026 and apply to the entire calendar year.",
        "expected_article": "Section A, Article A9.1.1"
    },
    {
        "question": "Within how many days must the FIA notify an applicant team of the outcome of its entry application?",
        "ground_truth": "The FIA will notify the applicant team of the outcome of the application within 30 days of the FIA's receipt of such application.",
        "expected_article": "Section A, Article A3.2.1"
    },
    {
        "question": "Can a decision by the Prosecuting Body to offer or enter into a settlement agreement be appealed?",
        "ground_truth": "There shall be no right of appeal in respect of any decision by the Prosecuting Body as to whether or not to offer or enter into a settlement agreement.",
        "expected_article": "Section A, Article A7.1.1"
    },

    # --- SECTION B: SPORTING REGULATIONS ---
    {
        "question": "What is the maximum speed limit in the pit lane during a Competition?",
        "ground_truth": "The speed limit in the pit lane will be 80 km/h, though it may be reduced to 60 km/h at specific events for safety reasons.",
        "expected_article": "Section B, Article B6.1"
    },
    {
        "question": "How many total sets of dry-weather tyres are allocated to each driver for a standard Competition weekend?",
        "ground_truth": "Each driver is allocated 13 sets of dry-weather tyres for a standard Competition weekend.",
        "expected_article": "Section B, Article B10.1"
    },
    {
        "question": "Under normal dry race conditions, what requirement exists regarding tyre compound usage during a Grand Prix?",
        "ground_truth": "Unless wet or intermediate tyres are used, each driver must use at least two different specifications of dry-weather tyres during the race.",
        "expected_article": "Section B, Article B10.5"
    },
    {
        "question": "How many points are awarded to the winner of a Grand Prix race?",
        "ground_truth": "25 points are awarded for 1st place in a Grand Prix race.",
        "expected_article": "Section B, Article B12.1"
    },
    {
        "question": "How many points are awarded to the winner of a Sprint Session?",
        "ground_truth": "8 points are awarded for 1st place in a Sprint Session.",
        "expected_article": "Section B, Article B12.2"
    },
    {
        "question": "What is the maximum distance or duration for a Sprint Session?",
        "ground_truth": "A Sprint Session is run over a distance of 100km or up to a maximum duration of 60 minutes.",
        "expected_article": "Section B, Article B3.5"
    },
    {
        "question": "What is the maximum overall time limit for a Grand Prix race including any race suspensions?",
        "ground_truth": "The maximum permitted time limit for a Grand Prix race, including all suspensions, is 3 hours.",
        "expected_article": "Section B, Article B20.1"
    },
    {
        "question": "What is the maximum continuous running time for a Grand Prix race excluding suspensions?",
        "ground_truth": "The maximum continuous running time for a Grand Prix race excluding suspensions is 2 hours.",
        "expected_article": "Section B, Article B20.2"
    },
    {
        "question": "When does Parc Fermé conditions begin for a car during a standard Competition weekend?",
        "ground_truth": "Parc Fermé conditions apply from the moment a car leaves the pit lane during Qualifying until the start of the race.",
        "expected_article": "Section B, Article B11.1"
    },
    {
        "question": "What penalty is imposed if a driver makes a false start (jumps the start)?",
        "ground_truth": "A false start is penalised by the Stewards with a 5-second time penalty, a 10-second time penalty, or a drive-through penalty depending on severity.",
        "expected_article": "Section B, Article B5.4"
    },
    {
        "question": "How many sets of intermediate tyres are allocated to each driver for a Competition weekend?",
        "ground_truth": "Each driver is allocated 4 sets of intermediate tyres for a Competition weekend.",
        "expected_article": "Section B, Article B10.2"
    },
    {
        "question": "How many sets of full wet tyres are allocated to each driver for a Competition weekend?",
        "ground_truth": "Each driver is allocated 3 sets of wet tyres for a Competition weekend.",
        "expected_article": "Section B, Article B10.3"
    },

    # --- SECTION C: TECHNICAL REGULATIONS ---
    {
        "question": "What is the minimum weight limit for the 2026 F1 car without fuel at all times during a Competition?",
        "ground_truth": "The minimum weight of the car without fuel must not be less than 768kg at all times during the Competition.",
        "expected_article": "Section C, Article C4.1"
    },
    {
        "question": "What is the minimum weight allocated for the driver including their seat and safety equipment?",
        "ground_truth": "The driver's weight including seat and safety equipment is fixed at a minimum of 80kg.",
        "expected_article": "Section C, Article C4.2"
    },
    {
        "question": "Is the MGU-H energy recovery unit permitted in the 2026 Power Unit regulations?",
        "ground_truth": "The MGU-H is eliminated and prohibited under the 2026 Power Unit regulations.",
        "expected_article": "Section C, Article C6.3"
    },
    {
        "question": "What is the maximum electrical power output limit for the MGU-K in the 2026 Power Unit?",
        "ground_truth": "The maximum electrical power output limit for the MGU-K is 350kW.",
        "expected_article": "Section C, Article C6.2"
    },
    {
        "question": "What type of fuel is mandated for use in the 2026 Formula 1 Power Units?",
        "ground_truth": "The fuel used in 2026 must be 100% advanced sustainable fuel certified by the FIA.",
        "expected_article": "Section C, Article C5.4"
    },
    {
        "question": "What is the maximum fuel mass flow rate permitted into the engine in 2026?",
        "ground_truth": "The maximum fuel mass flow rate limit is governed by the 3000 MJ/h energy flow limit formula.",
        "expected_article": "Section C, Article C5.1"
    },
    {
        "question": "What wheel rim diameter is specified for 2026 Formula 1 cars?",
        "ground_truth": "The wheel rim diameter for 2026 F1 cars is 18 inches.",
        "expected_article": "Section C, Article C10.1"
    },
    {
        "question": "Are active suspension systems permitted in 2026 F1 cars?",
        "ground_truth": "Active suspension systems are strictly prohibited.",
        "expected_article": "Section C, Article C8.1"
    },
    {
        "question": "What engine configuration is mandated for the 2026 internal combustion engine (ICE)?",
        "ground_truth": "The ICE must be a 1.6-litre V6 90-degree turbocharged four-stroke engine.",
        "expected_article": "Section C, Article C6.1"
    },
    {
        "question": "What active aerodynamics modes are introduced on the 2026 F1 car wings?",
        "ground_truth": "The 2026 regulations feature dual-mode active aerodynamics: Z-Mode (high downforce for cornering) and X-Mode (low drag for straights).",
        "expected_article": "Section C, Article C3.3"
    },
    {
        "question": "What is the maximum permitted compression ratio for the internal combustion engine cylinder?",
        "ground_truth": "The compression ratio of the internal combustion engine cylinders must not exceed 16.0:1.",
        "expected_article": "Section C, Article C6.1.4"
    },
    {
        "question": "How many forward gear ratios must the transmission gearbox have?",
        "ground_truth": "The transmission must have exactly 8 forward gear ratios.",
        "expected_article": "Section C, Article C7.1"
    },
    {
        "question": "Is reverse gear mandatory on all 2026 F1 cars?",
        "ground_truth": "Yes, all cars must have a functional reverse gear operable by the driver from the cockpit.",
        "expected_article": "Section C, Article C7.2"
    },
    {
        "question": "What safety feature is required above the driver's head to protect against cockpit intrusion?",
        "ground_truth": "The Halo cockpit protection structure is mandatory on all cars.",
        "expected_article": "Section C, Article C12.1"
    },

    # --- SECTION D: FINANCIAL REGULATIONS (F1 TEAMS) ---
    {
        "question": "What is the deadline for F1 Teams to submit their Full Year Reporting Documentation under the Cost Cap?",
        "ground_truth": "F1 Teams must submit their Full Year Reporting Documentation by 31 March of Year N+1.",
        "expected_article": "Section D, Article D1.2"
    },
    {
        "question": "Are driver salaries included within the F1 Team Cost Cap limit?",
        "ground_truth": "Driver salaries are an Excluded Cost and are not subject to the Cost Cap limit.",
        "expected_article": "Section D, Article D3.1"
    },
    {
        "question": "Are the salaries of the top 3 highest-paid personnel included in the F1 Team Cost Cap?",
        "ground_truth": "The consideration paid to the three highest-paid individual members of personnel are Excluded Costs.",
        "expected_article": "Section D, Article D3.1(b)"
    },
    {
        "question": "What percentage overspend defines a Minor Overspend Breach under the Financial Regulations for F1 Teams?",
        "ground_truth": "A Minor Overspend Breach occurs when a team exceeds the Cost Cap by less than 5.0%.",
        "expected_article": "Section D, Article D6.2"
    },
    {
        "question": "What defines a Major Overspend Breach under the Financial Regulations for F1 Teams?",
        "ground_truth": "A Major Overspend Breach occurs when an F1 Team exceeds the Cost Cap by 5.0% or more.",
        "expected_article": "Section D, Article D6.3"
    },
    {
        "question": "What body is responsible for monitoring compliance with the F1 Financial Regulations?",
        "ground_truth": "The Cost Cap Administration is responsible for administering, monitoring, and enforcing compliance with the Financial Regulations.",
        "expected_article": "Section D, Article D8.1"
    },
    {
        "question": "What type of penalty can be imposed for a Procedural Breach of the Financial Regulations?",
        "ground_truth": "A Procedural Breach results in a Financial Penalty and/or a Minor Sporting Penalty.",
        "expected_article": "Section D, Article D9.1"
    },
    {
        "question": "Are marketing and promotional activities included under the F1 Team Cost Cap?",
        "ground_truth": "Marketing and promotional costs are Excluded Costs under the Financial Regulations.",
        "expected_article": "Section D, Article D3.1(e)"
    },

    # --- SECTION E: FINANCIAL REGULATIONS (POWER UNIT MANUFACTURERS) ---
    {
        "question": "What is the Full Year Reporting deadline for Power Unit Manufacturers under Section E Financial Regulations?",
        "ground_truth": "Power Unit Manufacturers must submit their Full Year Reporting Documentation by 31 March of Year N+1.",
        "expected_article": "Section E, Article E1.2"
    },
    {
        "question": "Are costs associated with supplying Power Units to customer teams included under the PU Manufacturer Cost Cap?",
        "ground_truth": "Costs directly attributable to the supply of Power Units to customer teams are Excluded Costs.",
        "expected_article": "Section E, Article E3.1"
    },
    {
        "question": "What threshold defines a Minor Overspend Breach for a Power Unit Manufacturer?",
        "ground_truth": "A Minor Overspend Breach occurs when a PU Manufacturer exceeds the PU Cost Cap by less than 5.0%.",
        "expected_article": "Section E, Article E6.2"
    },
    {
        "question": "What threshold defines a Major Overspend Breach for a Power Unit Manufacturer?",
        "ground_truth": "A Major Overspend Breach occurs when a PU Manufacturer exceeds the PU Cost Cap by 5.0% or more.",
        "expected_article": "Section E, Article E6.3"
    },
    {
        "question": "Can a PU Manufacturer enter into an Accepted Breach Agreement (ABA) for a Material Overspend Breach?",
        "ground_truth": "No, an Accepted Breach Agreement cannot be entered into if a PU Manufacturer commits a Material (Major) Overspend Breach.",
        "expected_article": "Section E, Article E7.1"
    },

    # --- SECTION F: OPERATIONAL REGULATIONS ---
    {
        "question": "How long is the mandatory Summer Factory Shutdown period for F1 Teams?",
        "ground_truth": "The summer factory shutdown requires all F1 Teams to shut down operational activities for 14 consecutive calendar days during July and/or August.",
        "expected_article": "Section F, Article F3.1"
    },
    {
        "question": "What is the purpose of the Operational Curfew periods during a Competition weekend?",
        "ground_truth": "Curfew periods restrict team personnel from carrying out operational work on cars in the paddock during specified overnight rest windows.",
        "expected_article": "Section F, Article F2.1"
    },
    {
        "question": "Are teams allowed to run wind tunnel testing during the mandatory factory shutdown periods?",
        "ground_truth": "No, wind tunnel testing and CFD simulations are strictly prohibited during shutdown periods.",
        "expected_article": "Section F, Article F3.2"
    }
]

print(f"Generated {len(dataset)} Q&A pairs.")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Saved to {OUTPUT_JSON}")
