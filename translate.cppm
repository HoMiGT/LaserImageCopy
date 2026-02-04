module;

#include <QString>
#include <string>
#include <unicode/unistr.h>
#include <unicode/translit.h>
#include <unicode/utypes.h>
#include <stdexcept>

export module translate;


export std::string qStringToPinYin(const QString& input, bool removeSpaces = true);


inline bool isAsciiAlpha(const char ch) {
    const auto c = static_cast<unsigned char>(ch);
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
inline char toUpperAscii(const char ch) { return static_cast<char>(std::toupper(static_cast<unsigned char>(ch))); }
inline char toLowerAscii(const char ch) { return static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); }

static std::string titleCaseLatinRunsKeepOthers(const std::string& s, const bool removeSpaces) {
    std::string out;
    out.reserve(s.size());
    bool inWord = false;
    for (const char ch : s) {
        if (isAsciiAlpha(ch)) {
            out.push_back(inWord ? toLowerAscii(ch) : toUpperAscii(ch));
            inWord = true;
        }
        else {
            inWord = false;
            if (removeSpaces && ch == ' ') continue;
            out.push_back(ch); // 非字母原样保留
        }
    }
    return out;
}


std::string qStringToPinYin(const QString& input, bool removeSpaces) {
    UErrorCode status = U_ZERO_ERROR;

    // 关键点：直接用 UTF-16 构造 ICU UnicodeString（最稳）
    icu::UnicodeString u(reinterpret_cast<const UChar*>(input.utf16()), input.size());

    // auto* trans = icu::Transliterator::createInstance("Han-Latin", UTRANS_FORWARD, status);
    const auto* trans = icu::Transliterator::createInstance(
        "Han-Latin; NFD; [:Nonspacing Mark:] Remove; NFC",
        UTRANS_FORWARD,
        status
    );
    if (U_FAILURE(status) || !trans) {
        throw std::runtime_error("Failed to create ICU Transliterator (Han-Latin).");
    }
    trans->transliterate(u);
    delete trans;

    std::string utf8;
    u.toUTF8String(utf8);

    return titleCaseLatinRunsKeepOthers(utf8, removeSpaces);

}