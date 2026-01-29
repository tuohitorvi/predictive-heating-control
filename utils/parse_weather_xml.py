# utils/parse_weather_xml.py
import xml.etree.ElementTree as ET
from typing import List, Dict
from urllib.parse import urlparse, parse_qs

def parse_weather_xml(xml_str: str) -> List[Dict]:
    try:
        ns = {
            'wfs': 'http://www.opengis.net/wfs/2.0',
            'om': 'http://www.opengis.net/om/2.0',
            'wml2': 'http://www.opengis.net/waterml/2.0'
        }

        root = ET.fromstring(xml_str)
        observations_by_time = {}

        for member in root.findall('.//wfs:member', ns):
            observed_property = member.find('.//om:observedProperty', ns)
            if observed_property is None or observed_property.get('{http://www.w3.org/1999/xlink}href') is None:
                continue

            href_string = observed_property.get('{http://www.w3.org/1999/xlink}href')
            try:
                parsed_url = urlparse(href_string)
                query_params = parse_qs(parsed_url.query)
                param_name = query_params.get('param', [None])[0]
                if param_name is None:
                    param_name = href_string.split('/')[-1]
            except Exception:
                param_name = href_string.split('/')[-1]

            points = member.findall('.//wml2:point', ns)
            for point in points:
                time_node = point.find('wml2:MeasurementTVP/wml2:time', ns)
                value_node = point.find('wml2:MeasurementTVP/wml2:value', ns)

                if time_node is not None and value_node is not None and time_node.text is not None:
                    timestamp = time_node.text
                    if timestamp not in observations_by_time:
                        observations_by_time[timestamp] = {'time': timestamp}
                    try:
                        observations_by_time[timestamp][param_name] = float(value_node.text)
                    except (ValueError, TypeError):
                        observations_by_time[timestamp][param_name] = value_node.text

        sorted_timestamps = sorted(observations_by_time.keys())
        return [observations_by_time[ts] for ts in sorted_timestamps]

    except Exception as e:
        print(f"An unexpected error occurred in parse_weather_xml: {e}")
        return [{"error": "An unexpected error occurred during parsing", "details": str(e)}]

