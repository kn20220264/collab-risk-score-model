import { IconClose } from "./icons";

function Modal({ icon: Icon, title, subtitle, badge, onClose, children }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} aria-label="Zatvori">
          <IconClose className="icon" />
        </button>

        <div className="modal-header">
          {Icon && <Icon className="modal-header-icon" />}
          <h3>{title}</h3>
          {badge && <span className={`modal-badge modal-badge-${badge.tone}`}>{badge.label}</span>}
        </div>
        {subtitle && <p className="modal-subtitle">{subtitle}</p>}

        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}

export default Modal;
