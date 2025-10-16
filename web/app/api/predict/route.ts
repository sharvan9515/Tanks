import { NextResponse } from "next/server";
import { PredictRequestSchema } from "@/lib/validation";

export async function POST(req: Request) {
  const json = await req.json().catch(() => null);
  const parsed = PredictRequestSchema.safeParse(json);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid payload" }, { status: 400 });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);

  try {
    const res = await fetch(`${process.env.ML_URL}/predict`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(parsed.data),
      signal: controller.signal
    });

    clearTimeout(timeout);
    if (!res.ok) {
      return NextResponse.json({ error: "ML upstream error" }, { status: 502 });
    }

    const data = await res.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    clearTimeout(timeout);
    return NextResponse.json({ error: "ML unavailable" }, { status: 504 });
  }
}
