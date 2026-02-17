"use client";

import { useState } from "react";

export default function Home() {
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setResultImage(null);

    try {
      // Send image to backend
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Failed to process image");
      }

      // Get the result image and display it
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setResultImage(imageUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8 bg-gray-950 text-white">
      <h1 className="text-3xl font-bold mb-8">Face Landmark Detector</h1>

      {/* Upload Button */}
      <label className="cursor-pointer bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-medium transition">
        {loading ? "Processing..." : "Upload Image"}
        <input
          type="file"
          accept="image/*"
          onChange={handleUpload}
          disabled={loading}
          className="hidden"
        />
      </label>

      {/* Error Message */}
      {error && (
        <p className="mt-4 text-red-400">{error}</p>
      )}

      {/* Result Image */}
      {resultImage && (
        <div className="mt-8">
          <img
            src={resultImage}
            alt="Face with landmarks"
            className="max-w-lg rounded-lg shadow-lg"
          />
        </div>
      )}
    </main>
  );
}
