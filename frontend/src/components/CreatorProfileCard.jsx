import { useState } from "react";
import Modal from "./Modal";
import {
  IconEye,
  IconShield,
  IconUser,
  IconChat,
  IconBag,
  IconChart,
  IconHealth,
  IconWarning,
  IconClock,
} from "./icons";

const RISK_TONE = {
  "Nizak rizik": { tone: "low", label: "Safe to Partner", color: "#16a34a" },
  "Srednji rizik": { tone: "medium", label: "Umjeren rizik", color: "#d97706" },
  "Visok rizik": { tone: "high", label: "Visok rizik", color: "#dc2626" },
};

function formatCompact(n) {
  if (n == null) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}K`;
  return String(n);
}

function initials(name) {
  return (name || "?")
    .split(/\s+/)
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

function ScoreRing({ score, color }) {
  const r = 80;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - score / 100);

  return (
    <svg viewBox="0 0 200 200" className="profile-ring">
      <circle cx="100" cy="100" r={r} fill="none" stroke="#e5e7eb" strokeWidth="14" />
      <circle
        cx="100"
        cy="100"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="14"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        transform="rotate(-90 100 100)"
        className="profile-ring-fill"
      />
    </svg>
  );
}

function StatTile({ value, label }) {
  return (
    <div className="stat-tile">
      <div className="stat-tile-value">{value}</div>
      <div className="stat-tile-label">{label}</div>
    </div>
  );
}

function DetailTile({ icon: Icon, label, value, sublabel, onClick }) {
  return (
    <button className="detail-tile" onClick={onClick}>
      <div className="detail-tile-top">
        <span className="detail-tile-label">{label}</span>
        <Icon className="detail-tile-icon" />
      </div>
      <div className="detail-tile-value">{value}</div>
      <div className="detail-tile-sublabel">{sublabel}</div>
    </button>
  );
}

function ProgressBar({ pct, color }) {
  return (
    <div className="progress-track">
      <div className="progress-fill" style={{ width: `${Math.min(pct, 100)}%`, background: color }} />
    </div>
  );
}

const MODULE_LABELS = {
  quantitative: "Kvantitativne metrike",
  authenticity: "Autentičnost",
  sentiment: "Sentiment",
  brand_fit: "Brand-fit",
};

function ModuleScoreBar({ label, value, weight }) {
  return (
    <div className="profile-module-row">
      <div className="profile-module-label">
        {label}
        {weight != null && <span className="profile-module-weight">{Math.round(weight * 100)}%</span>}
      </div>
      <div className="profile-module-track">
        <div className="profile-module-fill" style={{ width: `${value}%` }} />
      </div>
      <div className="profile-module-value">{value}</div>
    </div>
  );
}

function CreatorProfileCard({ creator }) {
  const [activeModal, setActiveModal] = useState(null);

  const risk = RISK_TONE[creator.risk_category] || RISK_TONE["Srednji rizik"];
  const persona = creator.creator_persona || {};
  const profanity = creator.profanity_analysis || {};
  const brands = creator.brand_partners || {};
  const health = creator.audience_health || {};
  const contentAnalyzed = creator.content_analyzed || {};

  const tags = [persona.primary_niche, ...(persona.secondary_niches || []), ...(persona.content_style || [])].filter(
    Boolean
  );

  const profanityTone = profanity.level === "None" || profanity.level === "Low" ? "#16a34a" : "#dc2626";

  return (
    <div className="profile-card">
      <div className="profile-eyebrow">
        <IconEye className="icon" /> BRAND VETTING VIEW
      </div>

      <div className="profile-grid">
        <div className="profile-main">
          <div className="profile-identity">
            {creator.thumbnail_url ? (
              <img src={creator.thumbnail_url} alt={creator.display_name} className="profile-avatar" />
            ) : (
              <div className="profile-avatar profile-avatar-fallback">{initials(creator.display_name)}</div>
            )}
            <div>
              <h3 className="profile-name">{creator.display_name}</h3>
              <div className="profile-handle">{creator.handle}</div>
            </div>
          </div>

          <div className="profile-badges">
            <span className={`badge badge-${risk.tone}`}>
              <IconShield className="icon" /> {risk.label}
            </span>
            <span className="badge badge-score">{creator.score}/100</span>
          </div>

          {tags.length > 0 && (
            <div className="profile-tags">
              {tags.map((tag, i) => (
                <span className="tag-pill" key={i}>
                  {tag}
                </span>
              ))}
            </div>
          )}

          {persona.summary && <blockquote className="profile-quote">"{persona.summary}"</blockquote>}

          {creator.modules && (
            <div className="profile-modules">
              <div className="profile-modules-title">Skorovi po modulima</div>
              {Object.entries(MODULE_LABELS).map(([key, label]) => (
                <ModuleScoreBar
                  key={key}
                  label={label}
                  value={creator.modules[key]}
                  weight={creator.weights_used?.[key]}
                />
              ))}
            </div>
          )}

          {creator.ai_explanation && (
            <div className="profile-ai-note">
              <div className="profile-ai-note-title">AI obrazloženje</div>
              <p>{creator.ai_explanation}</p>
            </div>
          )}
        </div>

        <div className="profile-side">
          <div className="profile-ring-wrap">
            <ScoreRing score={creator.score} color={risk.color} />
            <div className="profile-ring-center">
              <span className="profile-ring-score">{creator.score}</span>
              <span className="profile-ring-caption">CREATORSCORE</span>
              <span className="profile-ring-tier" style={{ color: risk.color }}>
                {creator.tier}
              </span>
            </div>
          </div>

          <div className="stat-tile-grid">
            <StatTile value={formatCompact(creator.subscriber_count)} label="Total Reach" />
            <StatTile value={`${creator.engagement_rate_pct}%`} label="Engagement" />
            <StatTile value={contentAnalyzed.platforms ?? 1} label="Platforms" />
            <StatTile value={creator.triggered_risk_flags.length} label="Risk Flags" />
          </div>
        </div>
      </div>

      {creator.triggered_risk_flags.length > 0 && (
        <div className="profile-flags">
          <IconWarning className="icon" />
          <div>
            <div className="profile-flags-title">Kritični nalazi</div>
            {creator.triggered_risk_flags.map((flag, i) => (
              <div key={i} className="profile-flags-item">
                {flag}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="detail-tile-grid">
        <DetailTile
          icon={IconUser}
          label="Creator Persona"
          value={persona.primary_niche || "N/A"}
          sublabel="Content niche & brand fit"
          onClick={() => setActiveModal("persona")}
        />
        <DetailTile
          icon={IconChat}
          label="Profanity Level"
          value={profanity.level || "N/A"}
          sublabel={`${profanity.posts_with_profanity ?? 0}/${profanity.posts_analyzed ?? 0} posts flagged`}
          onClick={() => setActiveModal("profanity")}
        />
        <DetailTile
          icon={IconBag}
          label="Brand Partners"
          value={`${brands.total_brands ?? 0} Brands`}
          sublabel={`${brands.organic_count ?? 0} organic, ${brands.sponsored_count ?? 0} sponsored`}
          onClick={() => setActiveModal("brands")}
        />
        <DetailTile
          icon={IconChart}
          label="Content Analyzed"
          value={`${contentAnalyzed.posts_analyzed ?? 0} Posts`}
          sublabel={`${contentAnalyzed.comments_analyzed ?? 0} comments`}
          onClick={() => setActiveModal("content")}
        />
        <DetailTile
          icon={IconHealth}
          label="Audience Health"
          value={health.health_label || "N/A"}
          sublabel={`${health.bot_activity_pct ?? 0}% bot activity, ${health.toxic_pct ?? 0}% toxic`}
          onClick={() => setActiveModal("health")}
        />
      </div>

      <div className="profile-footer">
        <IconClock className="icon" /> Analizirano upravo sada
      </div>

      {activeModal === "persona" && (
        <Modal icon={IconUser} title="Creator Persona" subtitle="Content niche and style analysis" onClose={() => setActiveModal(null)}>
          {persona.summary && <div className="modal-info-box">{persona.summary}</div>}

          <div className="modal-stat-grid">
            <div className="modal-stat">
              <span className="modal-stat-label">Primary Niche</span>
              <span className="modal-stat-value">{persona.primary_niche || "N/A"}</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Secondary Niches</span>
              <span className="modal-stat-value">
                {persona.secondary_niches?.length ? persona.secondary_niches.join(", ") : "—"}
              </span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Content Style</span>
              <span className="modal-stat-value">
                {persona.content_style?.length ? persona.content_style.join(", ") : "—"}
              </span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Confidence</span>
              <span className="modal-stat-value">{Math.round((persona.confidence || 0) * 100)}%</span>
            </div>
          </div>

          {persona.context_for_risk && (
            <div className="modal-callout">
              <div className="modal-callout-title">Context for Risk Assessment</div>
              <p>{persona.context_for_risk}</p>
            </div>
          )}
        </Modal>
      )}

      {activeModal === "profanity" && (
        <Modal
          icon={IconChat}
          title="Profanity Level"
          subtitle="Explicit language usage across analyzed content"
          badge={{ label: profanity.level || "N/A", tone: profanity.level === "None" ? "low" : "high" }}
          onClose={() => setActiveModal(null)}
        >
          <div className="modal-stat-grid">
            <div className="modal-stat">
              <span className="modal-stat-label">Total Instances</span>
              <span className="modal-stat-value">{profanity.total_instances ?? 0}</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Posts Analyzed</span>
              <span className="modal-stat-value">{profanity.posts_analyzed ?? 0}</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Posts with Profanity</span>
              <span className="modal-stat-value">{profanity.posts_with_profanity ?? 0}</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Rate</span>
              <span className="modal-stat-value">{profanity.profanity_rate_pct ?? 0}%</span>
            </div>
          </div>

          <ProgressBar pct={profanity.profanity_rate_pct ?? 0} color={profanityTone} />

          <div className={`modal-note modal-note-${profanity.level === "None" ? "positive" : "warning"}`}>
            {profanity.summary}
          </div>
        </Modal>
      )}

      {activeModal === "brands" && (
        <Modal icon={IconBag} title="Brand Partnerships" subtitle="Brands mentioned across analyzed content" onClose={() => setActiveModal(null)}>
          <div className="modal-stat-grid modal-stat-grid-3">
            <div className="modal-stat">
              <span className="modal-stat-label">Total Brands</span>
              <span className="modal-stat-value">{brands.total_brands ?? 0}</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Organic</span>
              <span className="modal-stat-value modal-stat-organic">{brands.organic_count ?? 0}</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Sponsored</span>
              <span className="modal-stat-value modal-stat-sponsored">{brands.sponsored_count ?? 0}</span>
            </div>
          </div>

          {brands.top_brands?.length > 0 && (
            <>
              <div className="modal-section-title">Top Brands</div>
              <div className="modal-pill-row">
                {brands.top_brands.map((b, i) => (
                  <span key={i} className={`brand-pill brand-pill-${b.type === "sponsored" ? "sponsored" : "organic"}`}>
                    {b.name} ({b.mentions})
                  </span>
                ))}
              </div>
            </>
          )}

          {brands.categories?.length > 0 && (
            <>
              <div className="modal-section-title">Categories</div>
              <div className="modal-pill-row">
                {brands.categories.map((c, i) => (
                  <span key={i} className="brand-pill">
                    {c.name} ({c.count})
                  </span>
                ))}
              </div>
            </>
          )}
        </Modal>
      )}

      {activeModal === "content" && (
        <Modal icon={IconChart} title="Content Analyzed" subtitle="Data collected for score calculation" onClose={() => setActiveModal(null)}>
          <div className="modal-stat-grid modal-stat-grid-3">
            <div className="modal-stat">
              <span className="modal-stat-label">Posts</span>
              <span className="modal-stat-value">{contentAnalyzed.posts_analyzed ?? 0}</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Comments</span>
              <span className="modal-stat-value">{contentAnalyzed.comments_analyzed ?? 0}</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Platforms</span>
              <span className="modal-stat-value">{contentAnalyzed.platforms ?? 1}</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Flags</span>
              <span className="modal-stat-value">{creator.triggered_risk_flags.length}</span>
            </div>
          </div>

          <div className="modal-section-title">By Platform</div>
          <div className="modal-platform-row">
            <span>YouTube {creator.handle}</span>
            <span>
              {contentAnalyzed.posts_analyzed ?? 0} posts · {contentAnalyzed.comments_analyzed ?? 0} comments
            </span>
          </div>
        </Modal>
      )}

      {activeModal === "health" && (
        <Modal icon={IconHealth} title="Audience Health" subtitle="Bot & spam analysis of analyzed comments" onClose={() => setActiveModal(null)}>
          <div className="modal-stat-grid">
            <div className="modal-stat">
              <span className="modal-stat-label">Bot Activity</span>
              <span className="modal-stat-value">{health.bot_activity_pct ?? 0}%</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Toxic Comments</span>
              <span className="modal-stat-value">{health.toxic_pct ?? 0}%</span>
            </div>
            <div className="modal-stat">
              <span className="modal-stat-label">Overall</span>
              <span className="modal-stat-value">{health.health_label || "N/A"}</span>
            </div>
          </div>

          {health.sentiment_breakdown && (
            <>
              <div className="modal-section-title">Comment Sentiment</div>
              <div className="sentiment-bar">
                <div className="sentiment-bar-positive" style={{ width: `${health.sentiment_breakdown.positive_pct}%` }} />
                <div className="sentiment-bar-neutral" style={{ width: `${health.sentiment_breakdown.neutral_pct}%` }} />
                <div className="sentiment-bar-negative" style={{ width: `${health.sentiment_breakdown.negative_pct}%` }} />
              </div>
              <div className="sentiment-legend">
                <span className="sentiment-legend-positive">{health.sentiment_breakdown.positive_pct}% positive</span>
                <span className="sentiment-legend-neutral">{health.sentiment_breakdown.neutral_pct}% neutral</span>
                <span className="sentiment-legend-negative">{health.sentiment_breakdown.negative_pct}% negative</span>
              </div>
            </>
          )}
        </Modal>
      )}
    </div>
  );
}

export default CreatorProfileCard;
