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

data = {
    "model": "",
    "messages": [
        {
            "role": "user",
            "content": "write an essay on Llama",
        }
    ],
    "temperature": 0.1,
    "max_tokens": 50,
}

response = requests.post(url="http://127.0.0.1:8000/v1/chat/completions", json=data)
print(response.json())
