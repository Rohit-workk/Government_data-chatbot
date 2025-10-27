# datagov/client.py
import requests
import pandas as pd

class DataGovIndia:
    """Client for data.gov.in API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.data.gov.in/resource"
    
    def get_data(self, resource_id, filters=None, fields=None, limit=1000, offset=0):
        """
        Fetch data from data.gov.in API
        
        Args:
            resource_id: Dataset resource ID
            filters: Dictionary of filter conditions
            fields: List of fields to retrieve
            limit: Number of records to fetch
            offset: Offset for pagination
        
        Returns:
            pandas.DataFrame or None
        """
        
        params = {
            "api-key": self.api_key,
            "format": "json",
            "offset": offset,
            "limit": limit
        }
        
        # Add filters
        if filters:
            for key, value in filters.items():
                params[f"filters[{key}]"] = value
        
        # Add fields
        if fields:
            params["fields"] = ",".join(fields)
        
        try:
            url = f"{self.base_url}/{resource_id}"
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "records" in data:
                return pd.DataFrame(data["records"])
            else:
                return pd.DataFrame(data)
                
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
