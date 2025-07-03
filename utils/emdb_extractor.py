import argparse
import requests
import json
import sys

def extract_map_info_from_api(emdb_id):
    """
    Extract contour level and resolution from EMDB API
    
    Args:
        emdb_id (str): EMDB ID (e.g., 'EMD-1234' or '1234')
    
    Returns:
        tuple: (contour_level, resolution) or (None, None) if failed
    """
    # Clean EMDB ID - remove 'EMD-' prefix if present
    emdb_number = emdb_id.replace('EMD-', '').replace('emd-', '')
    api_url = f"https://www.ebi.ac.uk/emdb/api/entry/{emdb_number}"
    
    try:
        print(f"Fetching data for EMDB ID: EMD-{emdb_number}")
        print(f"API URL: {api_url}")
        
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extract contour level
        contour_level = data['map']['contour_list']['contour'][0]['level']
        
        # Extract resolution
        resolution = data['structure_determination_list']['structure_determination'][0]\
                    ['image_processing'][0]['final_reconstruction']['resolution']['valueOf_']
        
        return float(contour_level), float(resolution)
        
    except requests.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None, None
    except (KeyError, IndexError) as e:
        print(f"Error extracting data from JSON: {e}")
        return None, None
    except ValueError as e:
        print(f"Error converting values to float: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(
        description='Extract contour level and resolution from EMDB API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python emdb_extractor.py --emdb_id EMD-15635
        python emdb_extractor.py --emdb_id 15635
                """
    )
    
    parser.add_argument(
        '--emdb_id', 
        type=str, 
        required=True,
        help='EMDB ID (e.g., EMD-15635 or 15635)'
    )
    
    args = parser.parse_args()
    
    contour_level, resolution = extract_map_info_from_api(args.emdb_id)
    
    if contour_level is not None and resolution is not None:
        print(f"Contour Level: {contour_level}")
        print(f"Resolution: {resolution}")
            
    else:
        print(f"\nFailed to extract information for EMDB ID: {args.emdb_id}")
        sys.exit(1)

if __name__ == '__main__':
    main()