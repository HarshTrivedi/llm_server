import requests
import json

def main():

    # This is just an example. Change as necessary.
    host = "http://general-cirrascale-16.reviz.ai2.in"
    port = 49173

    probe_size_limits = (500, 2500)
    probe_size_increments = 100

    probe_size = probe_size_limits[0]
    while True:
        print(f"Probing size {probe_size}")

        probe_size += probe_size_increments
        if probe_size>= probe_size_limits[1]:
            break

        prompt = " ".join((["a"]*probe_size))
        params = {"prompt": prompt, "min_length": 1, "max_length": 1}
        response = requests.get(host + ":" + str(port) + "/generate", params=params)
        if response.ok:
            model_name = response.json()["model_name"]
            print(f"Successful ({model_name})")
        else:
            print("Failed")

if __name__ == '__main__':
    main()
