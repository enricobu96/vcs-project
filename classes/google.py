"""
 GoogleAPI class
 ---------------
 This class take as constructor the token to invoke google apis using this sequence:
    1. We get the text we want to send to google
    2. We send the text with the token which is necessary to identify the user permissions on other smart devices
    3. Google API response is then ri-elaborate in a custom way so that the response is sent to the user using a display
"""


class GoogleAPI:


    def __init__(self):
        self.token = ""

    def _sendTextMessage(self, message: str) -> any:
        
        pass

    def _elaborateResponse(self, response: any) -> bool:
        return True

    def execute(self, message: str) -> bool:
        res = self._sendTextMessage(message)
        newRes = self._elaborateResponse(res)
        return newRes