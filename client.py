import requests

URL = "http://localhost/predict"
TEST_AUDIO_FILE = "test_audio/down.wav"

if __name__ == "__main__":
    # load audio file
    audio_file = open(TEST_AUDIO_FILE, "rb").read()

    # send HTTP request with audio file
    values = {"file": (TEST_AUDIO_FILE, audio_file, "audio/wav")}

    # post request
    response = requests.post(URL, files=values)

    # parse response
    parse_response = response.json()

    # print predicted keyword
    print(f"Predicted keyword: {parse_response['keyword']}")
