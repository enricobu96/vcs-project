"""
 
 GoogleAPI class
 
 ---------------
 This class take as constructor the token to invoke google apis using this sequence:
    1. We get the text we want to send to google
    2. We send the text with the token which is necessary to identify the user permissions on other smart devices
    3. Google API response is then ri-elaborate in a custom way so that the response is sent to the user using a display

 ---------------

    Part of this code as been taken and edited in order to execute a single command.

    Copyright (C) 2017 Google Inc.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    MODIFICATIONS
    - Source: https://github.com/googlesamples/assistant-sdk-python/tree/master/google-assistant-sdk/googlesamples/assistant/grpc#run-the-samples

    The original code allows to execute an entire conversation using terminal or a display. 
    The edited code allows to execute a single request using the terminal.

 ---------------
"""

import argparse
import sys
import os
import logging
import json

import click
import google.auth.transport.grpc
import google.auth.transport.requests
import google.oauth2.credentials

from google.assistant.embedded.v1alpha2 import (
    embedded_assistant_pb2,
    embedded_assistant_pb2_grpc
)


class GoogleAPI:


    def __init__(self):
        self.language_code = "en-US"
        self.device_model_id = "vcs-project-unipd-app"
        self.device_id = "vcs-project-unipd-app-01"
        self.conversation_state = None
        # Force reset of first conversation.
        self.is_new_conversation = True
        self.assistant = None
        self.deadline = 60 * 3 + 5
        logging.basicConfig(level=logging.INFO)


    def _invokeAssistant(self, text_query):

        def iter_assist_requests():
            config = embedded_assistant_pb2.AssistConfig(
                audio_out_config=embedded_assistant_pb2.AudioOutConfig(
                    encoding='LINEAR16',
                    sample_rate_hertz=16000,
                    volume_percentage=0,
                ),
                dialog_state_in=embedded_assistant_pb2.DialogStateIn(
                    language_code=self.language_code,
                    conversation_state=self.conversation_state,
                    is_new_conversation=self.is_new_conversation,
                ),
                device_config=embedded_assistant_pb2.DeviceConfig(
                    device_id=self.device_id,
                    device_model_id=self.device_model_id,
                ),
                text_query=text_query,
            )
            # Continue current conversation with later requests.
            self.is_new_conversation = False
            req = embedded_assistant_pb2.AssistRequest(config=config)
            # assistant_helpers.log_assist_request_without_audio(req)
            yield req

        text_response = None
        # html_response = None
        for resp in self.assistant.Assist(iter_assist_requests(), self.deadline):
            # assistant_helpers.log_assist_response_without_audio(resp)
            if resp.dialog_state_out.conversation_state:
                conversation_state = resp.dialog_state_out.conversation_state
                self.conversation_state = conversation_state
            if resp.dialog_state_out.supplemental_display_text:
                text_response = resp.dialog_state_out.supplemental_display_text
        return text_response #, html_response


    def _sendTextMessage(self, message: str) -> any:
        api_endpoint = "embeddedassistant.googleapis.com"
        credentials = os.path.join(click.get_app_dir('google-oauthlib-tool'),'credentials.json')

        # Load credentials from oAuth 2.0
        try:
            with open(credentials, 'r') as f:
                credentials = google.oauth2.credentials.Credentials(token=None, **json.load(f))
                http_request = google.auth.transport.requests.Request()
                credentials.refresh(http_request)
        except Exception as e:
            logging.error('Error loading credentials: %s', e)
            logging.error('Run google-oauthlib-tool to initialize new OAuth 2.0 credentials.')
            return

        # Open an authorized gRPC channel using the specified endpoint
        grpc_channel = google.auth.transport.grpc.secure_authorized_channel(
            credentials, http_request, api_endpoint)
        logging.info('Connecting to %s', api_endpoint)

        # Construct the assistant
        self.assistant = embedded_assistant_pb2_grpc.EmbeddedAssistantStub(
            grpc_channel
        )

        # Invoke the assistant and get the response text
        logging.info('<you> %s', message)
        response_text = self._invokeAssistant(text_query=message)
        if response_text:
            logging.info('<@assistant> %s', response_text)
        grpc_channel.close()
        return response_text

    def _elaborateResponse(self, response: any) -> bool:
        return response

    def execute(self, message: str) -> bool:
        res = self._sendTextMessage(message)
        newRes = self._elaborateResponse(res)
        return newRes


def main(args):
    gapi = GoogleAPI()
    gapi.execute(sys.argv[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("<message>", help="Message to send to Google Assistant")
    args = parser.parse_args()
    main(args)