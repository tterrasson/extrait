/**
 * Shared example environment helpers.
 *
 * `baseURL` is mandatory in extrait: there is no implicit vendor endpoint, so an
 * example started without `LLM_BASE_URL` fails here instead of sending the key
 * to a host nobody configured.
 */

export function requireBaseURL(): string {
  const baseURL = process.env.LLM_BASE_URL;

  if (!baseURL) {
    throw new Error(
      "LLM_BASE_URL is required. Point it at your endpoint, e.g. LLM_BASE_URL=http://localhost:1234/v1",
    );
  }

  return baseURL;
}
