import { z } from "zod";

export const PredictRequestSchema = z.object({
  features: z.array(z.number({ invalid_type_error: "Features must be numeric" })).min(1)
});

export type PredictRequest = z.infer<typeof PredictRequestSchema>;
