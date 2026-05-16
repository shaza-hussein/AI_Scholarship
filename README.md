# AI Scholarship Project

A Python-based scholarship data extraction project focused on collecting scholarship listings for selected fields of study. The repository includes web scrapers for Computer Science, Cybersecurity, Theology, and Web Design, with extracted scholarship data saved as JSON and CSV files.

## Project Overview

This project is designed to:
- Scrape scholarship directory pages for specific majors
- Extract detailed scholarship information such as name, deadline, award availability, eligibility tags, and application links
- Save the results in structured JSON format
- Provide a data foundation for later analysis or dataset generation

## Repository Structure

- `scrapers/`
  - `scraper_computer_science.py`
  - `scraper_cybersecurity.py`
  - `scraper_theology.py`
  - `scraper_web_design.py`
- `data/`
  - `europe_scholarships_detailed.csv`
  - `europe_scholarships_v2.csv`
  - `theology_scholarships_data.json`
  - `web_design_scholarships_data.json`
- `data_Json/`
  - `computer_science_scholarships_data.json`
  - `cybersecurity_scholarships_data.json`
  - `theology_scholarships_data.json`
  - `web_design_scholarships_data.json`
- `requirements.txt`
- `schema.json`

## Requirements

The project depends on the following Python packages:

- `beautifulsoup4==4.14.3`
- `certifi==2026.4.22`
- `charset-normalizer==3.4.7`
- `idna==3.13`
- `lxml==6.1.0`
- `numpy==2.4.4`
- `pandas==3.0.2`
- `python-dateutil==2.9.0.post0`
- `requests==2.33.1`
- `six==1.17.0`
- `soupsieve==2.8.3`
- `typing_extensions==4.15.0`
- `tzdata==2026.2`
- `urllib3==2.7.0`

## Installation

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Usage

Each scraper can be executed independently. Example:

```powershell
python scrapers\scraper_computer_science.py
```

Repeat the command for other scraper files to collect data for each major.

## Output

Each scraper writes a JSON file with extracted scholarship data. Example output file names:

- `computer_science_scholarships_data.json`
- `cybersecurity_scholarships_data.json`
- `theology_scholarships_data.json`
- `web_design_scholarships_data.json`

The existing `data/` and `data_Json/` directories contain previously generated scholarship datasets and project sample data.

## Notes

- The scrapers use `requests` and `BeautifulSoup` for HTML parsing.
- The scripts include delays between requests to reduce the risk of IP blocking.
- `schema.json` is available for data schema or validation; update it as needed to reflect the exported JSON structure.

## License

This repository does not include a specific license file. Add a LICENSE if you plan to share or publish the project publicly.
