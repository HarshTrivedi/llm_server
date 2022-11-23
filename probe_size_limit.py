import requests
import json

def main():

    # This is just an example. Change as necessary.
    host = "http://general-cirrascale-15.reviz.ai2.in"
    port = 49192

    probe_size_limits = (300, 4000)
    probe_size_increments = 100

    probe_size = probe_size_limits[0]
    while True:

        probe_size += probe_size_increments
        if probe_size>= probe_size_limits[1]:
            break

        print(f"Probing size {probe_size}")
        prompt = " ".join((["a"]*probe_size))
        params = {"prompt": prompt, "min_length": 1, "max_length": probe_size + 1}
        response = requests.get(host + ":" + str(port) + "/generate", params=params)
        if response.ok:
            model_name = response.json()["model_name"]
            print(response.json()["generated_texts"][0])
            num_generated_tokens = response.json()["generated_num_tokens"]
            run_time_in_seconds = response.json()["run_time_in_seconds"]
            print(f"Successful ({model_name}, {num_generated_tokens}, {run_time_in_seconds})")
        else:
            print("Failed")

if __name__ == '__main__':
    main()
