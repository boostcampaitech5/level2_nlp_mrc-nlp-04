import re
import sys
import json
import requests
import argparse

from requests.structures import CaseInsensitiveDict

def main(title):
	with open("body.txt", "r") as f:
		text = f.read()
	label = re.findall(r'###\s(.*)', text)
	value = {}
	paragraph = re.split(r"\n\s*\n", re.sub(r"###\s(.*)", "", text).strip())
	for i, cls in enumerate(label):
		value[cls] = paragraph[i]

	query_json = '''{
		    "parent": {
		        "database_id": "64a8016ef93f4829a090b9c0df0372cd"
		    },
		    "properties": {
		        "Base Model": {
		            "type": "relation",
		            "relation": [

		            ],
		            "has_more": false
		        },
		        "Github Issue": {
		            "type": "url",
		            "url": null
		        },
		        "읽은이": {
		            "type": "people",
		            "people": []
		        },
		        "담당자": {
		            "type": "people",
		            "people": []
		        },
		        "태그": {
		            "type": "multi_select",
		            "multi_select": []
		        },
		        "EM": {
		            "type": "number",
		            "number": 99
		        },
		        "F1": {
		            "type": "number",
		            "number": 99
		        },
		        "이름": {
		            "type": "title",
		            "title": [
		                {
		                    "type": "text",
		                    "text": {
		                        "content": "Test"
		                    }
		                }
		            ]
		        },
		        "현재 상태": {
					"type": "status",
					"status": {
						"id": "c35f48eb-6572-403f-a50c-be26e288f107",
                        "name": "Complete",
                        "color": "green"
					}
				}
		    },
		    "children": [
		        {
		            "object": "block",
		            "type": "heading_1",
		            "heading_1": {
		                "rich_text": [
		                    {
		                        "type": "text",
		                        "text": {
		                            "content": "Description"
		                        }
		                    }
		                ]
		            }
		        },
		        {
		            "object": "block",
		            "type": "paragraph",
		            "paragraph": {
		                "rich_text": [
		                    {
		                        "type": "text",
		                        "text": {
		                            "content": ""
		                        }
		                    }
		                ]
		            }
		        }
		    ]

		}'''

	query = json.loads(query_json)
	query['properties']['이름']['title'][0]['text']['content'] = title
	query['properties']['EM']['number'] = float(value['EM'])
	query['properties']['F1']['number'] = float(value['F1'])
	query['children'][1]['paragraph']['rich_text'][0]['text']['content'] = value['Description']
	query_json = json.dumps(query, indent=4, ensure_ascii=False)

	req_pages_create = "https://api.notion.com/v1/pages/"
	headers = CaseInsensitiveDict()
	headers["Authorization"] = "Bearer " + "secret_yQviW8ZA9wvn5j76fBRZvsQltgoVROi4D8NPCCXmej0"
	headers["Notion-Version"] = "2022-02-22"
	headers['Content-Type'] = 'application/json'
	res = requests.post(req_pages_create, data=query_json.encode('utf-8'), headers=headers)

	if res.status_code == 200:
		print("*"*4,"Success", "*"*4)
		print("Check Notion Page")
		return
	else:
		print("*"*4, "Fail", "*"*4)
		print(res.text)
		return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--title", required=True, type=str, default="Default Title")
	args = parser.parse_args()

	main(title=args.title)
