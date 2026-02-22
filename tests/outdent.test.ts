import { describe, expect, test } from "bun:test";
import { createOutdent } from "../src/outdent";

describe("createOutdent", () => {
  test("dedents multiline templates while preserving relative indentation", () => {
    const outdent = createOutdent({
      trimLeadingNewline: true,
      trimTrailingNewline: true,
      newline: "\n",
    });

    const result = outdent`
      root
        child
      sibling
    `;

    expect(result).toBe("root\n  child\nsibling");
  });

  test("preserves empty lines inside the template", () => {
    const outdent = createOutdent({
      trimLeadingNewline: true,
      trimTrailingNewline: true,
      newline: "\n",
    });

    const result = outdent`
      before

        middle

      after
    `;

    expect(result).toBe("before\n\n  middle\n\nafter");
  });

  test("normalizes CRLF and CR in template segments only", () => {
    const outdent = createOutdent({
      trimLeadingNewline: true,
      trimTrailingNewline: true,
      newline: "\n",
    });
    const interpolated = "A\r\nB";

    const result = outdent`
      start
      ${interpolated}
      end
    `;

    expect(result).toBe("start\nA\r\nB\nend");
  });

  test("can keep leading/trailing newline when trim is disabled", () => {
    const outdent = createOutdent({
      trimLeadingNewline: false,
      trimTrailingNewline: false,
      newline: "\n",
    });

    const result = outdent`
      line
    `;

    expect(result).toBe("\nline\n");
  });

  test("string() dedents lines after the first newline", () => {
    const outdent = createOutdent({
      trimLeadingNewline: false,
      trimTrailingNewline: false,
      newline: "\n",
    });

    const result = outdent.string("Header\n    line 1\n    line 2");

    expect(result).toBe("Header\nline 1\nline 2");
  });

  test("string() keeps original newline characters when normalization is disabled", () => {
    const outdent = createOutdent({
      trimLeadingNewline: false,
      trimTrailingNewline: false,
      newline: null,
    });

    const result = outdent.string("A\r\nB\rC\n");

    expect(result).toBe("A\r\nB\rC\n");
  });

  test("removes indentation by character count across mixed tabs/spaces", () => {
    const outdent = createOutdent({
      trimLeadingNewline: true,
      trimTrailingNewline: true,
      newline: "\n",
    });

    const result = outdent`
		alpha
	 	beta
	  gamma
    `;

    expect(result).toBe("alpha\n\tbeta\n gamma");
  });
});
