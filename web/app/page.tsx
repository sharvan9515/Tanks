"use client";

import { FormEvent, useState } from "react";

export default function Page() {
  const [rawFeatures, setRawFeatures] = useState("1, 2, 3");
  const [prediction, setPrediction] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setPrediction("");
    setError("");

    const features = rawFeatures
      .split(",")
      .map((value) => Number(value.trim()))
      .filter((value) => !Number.isNaN(value));

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ features })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error ?? "Unexpected error");
      }

      setPrediction(JSON.stringify(data, null, 2));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col gap-6 bg-white p-8 text-gray-900">
      <section>
        <h1 className="text-3xl font-semibold">Tank Cost Estimator</h1>
        <p className="mt-2 text-sm text-gray-600">
          Paste the numeric features exported by your engineering pipeline. Values should be comma-separated and in the
          exact order expected by the production model.
        </p>
      </section>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <label className="flex flex-col gap-2">
          <span className="text-sm font-medium text-gray-700">Feature vector</span>
          <textarea
            className="min-h-[120px] rounded border border-gray-300 p-3 font-mono text-sm focus:border-blue-500 focus:outline-none"
            value={rawFeatures}
            onChange={(event) => setRawFeatures(event.target.value)}
            placeholder="1.2, 3.4, 5.6"
          />
        </label>

        <button
          type="submit"
          className="inline-flex w-max items-center justify-center rounded bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-700 focus:outline-none focus:ring focus:ring-blue-300 disabled:cursor-not-allowed disabled:bg-blue-300"
          disabled={isLoading}
        >
          {isLoading ? "Predicting…" : "Run prediction"}
        </button>
      </form>

      {(prediction || error) && (
        <section className="rounded border border-gray-200 bg-gray-50 p-4">
          <h2 className="text-lg font-semibold">Result</h2>
          {error ? (
            <p className="mt-2 text-sm text-red-600">{error}</p>
          ) : (
            <pre className="mt-2 whitespace-pre-wrap text-sm text-gray-800">{prediction}</pre>
          )}
        </section>
      )}
    </main>
  );
}
