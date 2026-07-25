function icon(paths, viewBox = "0 0 24 24") {
  return function Icon({ className }) {
    return (
      <svg
        className={className}
        viewBox={viewBox}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {paths}
      </svg>
    );
  };
}

export const IconEye = icon(
  <>
    <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z" />
    <circle cx="12" cy="12" r="3" />
  </>
);

export const IconShield = icon(
  <path d="M12 2 4 5v6c0 5 3.5 8.5 8 10 4.5-1.5 8-5 8-10V5l-8-3Z" />
);

export const IconUser = icon(
  <>
    <circle cx="12" cy="8" r="4" />
    <path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7" />
  </>
);

export const IconChat = icon(
  <path d="M4 4h16v12H8l-4 4V4Z" />
);

export const IconBag = icon(
  <>
    <path d="M6 8h12l1 13H5L6 8Z" />
    <path d="M9 8V6a3 3 0 0 1 6 0v2" />
  </>
);

export const IconChart = icon(
  <>
    <path d="M4 20V10" />
    <path d="M11 20V4" />
    <path d="M18 20v-7" />
  </>
);

export const IconHealth = icon(
  <>
    <rect x="3" y="7" width="18" height="13" rx="2" />
    <path d="M9 7V5a3 3 0 0 1 6 0v2" />
    <path d="M8 13h3v3h2v-3h3v-2h-3V8h-2v3H8v2Z" />
  </>
);

export const IconWarning = icon(
  <>
    <path d="M12 3 2 20h20L12 3Z" />
    <path d="M12 10v4" />
    <path d="M12 17h.01" />
  </>
);

export const IconClose = icon(<path d="M5 5l14 14M19 5 5 19" />);

export const IconClock = icon(
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 3" />
  </>
);

export const IconCheck = icon(<path d="M4 12l5 5L20 6" />);

export const IconHeart = icon(
  <path d="M12 20.5s-7.5-4.6-10-9.2C.5 8 2 4.5 5.5 4c2-.3 3.7.6 4.9 2.1L12 7.8l1.6-1.7C14.8 4.6 16.5 3.7 18.5 4c3.5.5 5 4 3.5 7.3-2.5 4.6-10 9.2-10 9.2Z" />
);
