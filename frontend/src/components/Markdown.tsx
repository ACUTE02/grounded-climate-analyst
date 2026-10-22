import { Fragment, type ReactNode } from "react";

// A small, safe renderer for the report text the backend writes: headings, bold,
// bullet lists, block quotes, paragraphs. It builds React elements only (no raw HTML).

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  let last = 0;
  let key = 0;
  for (const m of text.matchAll(re)) {
    const i = m.index ?? 0;
    if (i > last) out.push(text.slice(last, i));
    const token = m[0];
    out.push(
      token.startsWith("**") ? (
        <strong key={key++}>{token.slice(2, -2)}</strong>
      ) : token.startsWith("*") ? (
        <em key={key++}>{token.slice(1, -1)}</em>
      ) : (
        <code key={key++}>{token.slice(1, -1)}</code>
      ),
    );
    last = i + token.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export default function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let quote: string[] = [];
  let para: string[] = [];
  let key = 0;

  const flush = () => {
    if (para.length) {
      blocks.push(<p key={key++}>{inline(para.join(" "))}</p>);
      para = [];
    }
    if (list.length) {
      blocks.push(
        <ul key={key++}>
          {list.map((item, i) => (
            <li key={i}>{inline(item)}</li>
          ))}
        </ul>,
      );
      list = [];
    }
    if (quote.length) {
      blocks.push(<blockquote key={key++}>{inline(quote.join(" "))}</blockquote>);
      quote = [];
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    const bullet = /^\s*[*-]\s+(.*)$/.exec(line);
    const quoted = /^>\s?(.*)$/.exec(line);
    if (!line.trim()) {
      flush();
    } else if (heading) {
      flush();
      const level = Math.min(heading[1].length + 1, 4);
      const Tag = (`h${level}` as unknown) as "h2";
      blocks.push(<Tag key={key++}>{inline(heading[2])}</Tag>);
    } else if (bullet) {
      if (para.length || quote.length) flush();
      list.push(bullet[1]);
    } else if (quoted) {
      if (para.length || list.length) flush();
      quote.push(quoted[1]);
    } else {
      if (list.length || quote.length) flush();
      para.push(line.trim());
    }
  }
  flush();
  return <Fragment>{blocks}</Fragment>;
}
