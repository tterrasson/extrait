interface ColorSupportConfig {
  colors: boolean;
}

const ANSI = {
  reset: "\u001b[0m",
  bold: "\u001b[1m",
  cyan: "\u001b[36m",
  yellow: "\u001b[33m",
  green: "\u001b[32m",
  red: "\u001b[31m",
  dim: "\u001b[2m",
} as const;

export function color(
  config: ColorSupportConfig,
  text: string,
  tone: "cyan" | "yellow" | "green" | "red",
): string {
  if (!config.colors) {
    return text;
  }

  return `${ANSI[tone]}${text}${ANSI.reset}`;
}

export function dim(config: ColorSupportConfig, text: string): string {
  if (!config.colors) {
    return text;
  }

  return `${ANSI.dim}${text}${ANSI.reset}`;
}

export function title(config: ColorSupportConfig, text: string): string {
  if (!config.colors) {
    return text;
  }

  return `${ANSI.bold}${text}${ANSI.reset}`;
}
