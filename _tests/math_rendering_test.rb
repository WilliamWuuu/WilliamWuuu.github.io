require 'minitest/autorun'
require 'yaml'
require 'cgi'
require 'kramdown'
require 'kramdown-parser-gfm'

class MathRenderingTest < Minitest::Test
  ROOT = File.expand_path('..', __dir__)
  OPTIONS = YAML.load_file(File.join(ROOT, '_config.yml'))
                .fetch('kramdown').each_with_object({}) { |(key, value), result| result[key.to_sym] = value }
                .merge(syntax_highlighter: nil)

  def render(source)
    Kramdown::Document.new(source, OPTIONS).to_html
  end

  def math_sources(html)
    html.scan(%r{<(span|div)\b[^>]*class="math-source"[^>]*>(.*?)</\1>}m)
        .map { |tag, source| [tag, CGI.unescapeHTML(source)] }
  end

  def test_inline_math_does_not_consume_surrounding_text_or_emphasis
    source = <<~'MARKDOWN'
      abc <span class="math-source" markdown="0">\(v^*\)</span> efg <span class="math-source" markdown="0">\(q^*\)</span> ijk. *Normal emphasis*.
      Also <span class="math-source" markdown="0">\(\mathbb{E}_\pi\)</span> and <span class="math-source" markdown="0">\(v_\pi\)</span>.
    MARKDOWN
    html = render(source)

    assert_equal math_sources(source), math_sources(html)
    assert_includes html, '</span> efg <span'
    assert_equal ['Normal emphasis'], html.scan(%r{<em>(.*?)</em>}).flatten
  end

  def test_display_math_in_a_proof_preserves_alignment_and_html_characters
    source = <<~'MARKDOWN'
      <details class="proof" markdown="1">
      <summary>Proof</summary>

      <div class="math-source" markdown="0">
      \[
      \begin{aligned}
      x_i &amp;= a &lt; b \\
      y^* &amp;= c &gt; d
      \end{aligned}
      \]
      </div>

      *Explanation* with <span class="math-source" markdown="0">\(x_i\)</span>.

      </details>
    MARKDOWN
    html = render(source)

    assert_equal math_sources(source), math_sources(html)
    assert_includes html, '<details class="proof">'
    assert_includes html, '<em>Explanation</em>'
    refute_match %r{<br\s*/?>}, html
  end

  def test_code_examples_keep_their_delimiters_literal
    html = render(<<~'MARKDOWN')
      Inline code: `\(v^*\)` and `$v^*$`.

      ```tex
      \[x_i & y^*\]
      $$z$$
      ```
    MARKDOWN

    assert_includes html, '<code>\(v^*\)</code>'
    assert_includes html, '<code>$v^*$</code>'
    assert_includes CGI.unescapeHTML(html), '\[x_i & y^*\]'
    assert_includes html, '$$z$$'
    assert_empty math_sources(html)
  end

  def test_dollar_math_is_not_converted_to_katex_delimiters
    html = render("Old $x_i$ and $$y^*$$.\n\n$$\nz^*\n$$\n")

    refute_includes html, '\('
    refute_includes html, '\['
    assert_includes html, '$x_i$'
    assert_includes html, 'kdmath'
  end

  def test_every_published_formula_survives_markdown_intact
    files = Dir[File.join(ROOT, '_blogs', '**', '*.md')] +
            Dir[File.join(ROOT, '_showcase', '**', '*.md')]
    total = 0

    files.each do |file|
      source = File.read(file).sub(/\A---\s*\n.*?\n---\s*\n/m, '')
      expected = math_sources(source)
      next if expected.empty?

      assert_equal expected, math_sources(render(source)), file
      expected.each do |tag, formula|
        trimmed = formula.strip
        opening, closing = tag == 'span' ? ['\(', '\)'] : ['\[', '\]']
        assert trimmed.start_with?(opening) && trimmed.end_with?(closing), file
      end
      total += expected.size
    end

    assert_operator total, :>, 0
  end
end
