import { jsonrepair } from "jsonrepair";
import type { ExtractionParseHint } from "./types";
import { toErrorMessage } from "./utils/common";

export function buildParseHint(
  content: string,
  allowRepair: boolean,
  hintMaxLength: number,
): ExtractionParseHint | null {
  if (content.length > hintMaxLength) {
    return null;
  }

  try {
    return {
      success: true,
      parsed: JSON.parse(content),
      repaired: null,
      usedRepair: false,
      stage: "parse",
      error: "",
    };
  } catch (directError) {
    if (!allowRepair) {
      return {
        success: false,
        parsed: null,
        repaired: null,
        usedRepair: false,
        stage: "parse",
        error: toErrorMessage(directError),
      };
    }

    let repaired: string;
    try {
      repaired = jsonrepair(content);
    } catch (repairError) {
      return {
        success: false,
        parsed: null,
        repaired: null,
        usedRepair: true,
        stage: "repair",
        error: toErrorMessage(repairError),
      };
    }

    try {
      return {
        success: true,
        parsed: JSON.parse(repaired),
        repaired,
        usedRepair: true,
        stage: "parse",
        error: "",
      };
    } catch (parseError) {
      return {
        success: false,
        parsed: null,
        repaired,
        usedRepair: true,
        stage: "parse",
        error: toErrorMessage(parseError || directError),
      };
    }
  }
}

export function parseHintBonus(hint: ExtractionParseHint): number {
  if (hint.success) {
    return hint.usedRepair ? 70 : 120;
  }

  return hint.usedRepair ? -20 : -10;
}
