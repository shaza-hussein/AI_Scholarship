import requests
from bs4 import BeautifulSoup
import time
import json
import random
import re

# 1. Fake browser settings to avoid IP blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

session = requests.Session()
session.headers.update(HEADERS)

def get_scholarship_links():
    """
    Phase 1: Collect scholarship links from the Computer Science directory table.
    """
    print(">>> Starting Phase 1: Collecting Computer Science scholarship links...")
    scholarship_urls = []
    
    # Specific URL for the Computer Science major
    search_url = "https://www.scholarships.com/financial-aid/college-scholarships/scholarship-directory/academic-major/computer-science"
    
    try:
        response = session.get(search_url, timeout=10)
        
        if response.status_code != 200:
            print(f"Warning: Website refused connection! Status Code: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract links from the table rows
        link_elements = soup.select('td a.blacklink')
        
        if not link_elements:
            print("No links found in the directory table.")
            return []
            
        for a_tag in link_elements:
            href = a_tag.get('href')
            if href:
                full_link = f"https://www.scholarships.com{href}"
                if full_link not in scholarship_urls:
                    scholarship_urls.append(full_link)
        
        print(f"Successfully found {len(scholarship_urls)} links.")
        
    except Exception as e:
        print(f"Error occurred while fetching the list: {e}")
            
    print(f"\nLinks collected and ready for Phase 2 extraction!\n")
    return scholarship_urls

def extract_scholarship_data(url):
    """
    Phase 2: Extract detailed data including tags, award counts, and geographical location.
    """
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Extract Scholarship Name
        name_elem = soup.find('h1')
        scholarship_name = name_elem.text.strip() if name_elem else "N/A"
        
        # 2. Extract Amount and Deadline
        h5_tags = soup.find_all('h5')
        funding_type = h5_tags[0].text.strip() if len(h5_tags) >= 1 else "N/A"
        deadline = h5_tags[1].text.strip() if len(h5_tags) >= 2 else "N/A"

        # 3. Extract Award Availability and Metadata Tags
        awards_available = "N/A"
        tags = []

        awards_span = soup.find('span', string=lambda text: text and 'Awards Available:' in text)
        if awards_span and awards_span.parent:
            awards_h5 = awards_span.parent.find_next_sibling('h5')
            if awards_h5:
                awards_available = awards_h5.text.strip()

        tags_heading = soup.find('h5', string=lambda text: text and ('Qualified Based On:' in text or 'Eligibility Criteria:' in text))
        if tags_heading:
            tags_container = tags_heading.find_next_sibling('div')
            if tags_container:
                tags = [tag.text.strip() for tag in tags_container.find_all('div')]

        # 4. Professional extraction of the 4 main text sections
        description, details, eligibility, application_process = "N/A", "N/A", "N/A", "N/A"
        
        main_desc_heading = soup.find('h2', string=lambda text: text and 'Scholarship Description' in text)
        if main_desc_heading:
            content_div = main_desc_heading.find_next_sibling('div')
            if content_div:
                first_p = content_div.find('p')
                if first_p: description = first_p.get_text(strip=True)
                
                det_h2 = content_div.find('h2', string=lambda text: text and 'Details' in text)
                if det_h2: details = det_h2.find_next_sibling().get_text(separator="\n", strip=True)
                
                elig_h2 = content_div.find('h2', string=lambda text: text and 'Eligibility' in text)
                if elig_h2: eligibility = elig_h2.find_next_sibling().get_text(separator="\n", strip=True)
                
                app_h2 = content_div.find('h2', string=lambda text: text and 'Application Process' in text)
                if app_h2: application_process = app_h2.find_next_sibling().get_text(separator="\n", strip=True)

        # 5. Strict Geographic Location Search (Regex to avoid copyright text)
        location = "USA"
        contact_div = soup.find('div', class_='contactDetail')
        
        if contact_div:
            lines = [l.strip() for line in contact_div.stripped_strings for l in line.split('\n') if l.strip()]
            for line in lines:
                # Matches patterns like: City, ST 12345
                if re.search(r', [A-Z]{2} \d+', line):
                    location = f"{line} (USA)"
                    break
        
        # Fallback: search full page if Contact section is missing
        if location == "USA":
            all_lines = [l.strip() for l in soup.get_text(separator="\n").split('\n') if l.strip()]
            for line in all_lines:
                if len(line) < 100 and re.search(r', [A-Z]{2} \d+', line):
                    location = f"{line} (USA)"
                    break

        # 6. Final JSON data structure
        data = {
            "scholarship_name": scholarship_name,
            "country": location, 
            "degree_level": "Computer Science", 
            "funding_type": funding_type,
            "awards_available": awards_available,
            "deadline": deadline,
            "tags": tags,
            "description": description,
            "scholarship_details": details,
            "eligibility": eligibility,
            "application_process": application_process,
            "application_link": url
        }
        return data

    except Exception as e:
        print(f"Failed to extract data for URL {url}: {e}")
        return None

def main():
    urls = get_scholarship_links()
    results = []
    
    if not urls:
        print("No links found to begin Phase 2. Shutting down.")
        return

    print(">>> Starting Phase 2: Extracting details (this will take time to ensure no blocking)...")
    
    for index, url in enumerate(urls, 1):
        print(f"[{index}/{len(urls)}] Processing scholarship...")
        data = extract_scholarship_data(url)
        
        if data:
            results.append(data)
            
        # Crucial delay to avoid IP banning
        time.sleep(random.uniform(1.5, 3.0)) 
        
    print("\n>>> Saving data to file...")
    filename = 'computer_science_scholarships_data.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Success! Saved {len(results)} scholarships to '{filename}'")

if __name__ == "__main__":
    main()