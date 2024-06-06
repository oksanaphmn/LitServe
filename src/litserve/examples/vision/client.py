# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import requests
import base64

# Read the image file and encode it to base64
with open("dog.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# Send the POST request to the server
response = requests.post("http://127.0.0.1:8000/predict", json={"image_data": encoded_string})

# Print the response from the server
print(response.json())
