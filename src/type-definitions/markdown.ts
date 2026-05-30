export interface MarkdownCodeBlock {
  language: string | null;
  code: string;
  start: number;
  end: number;
}

export interface MarkdownCodeOptions {
  language?: string;
  firstOnly?: boolean;
}
