import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from slsim.ImageSimulation.image_simulation import simulate_image, rgb_image_from_image_list
from tqdm import tqdm

def make_multiband_images_and_rgb_image(
    lens_class,
    bands=["g", "r", "i", "z", "y"],
    num_pix=41,
    coadd_years=10,
    add_noise=True,
    rgb_bands=["i", "r", "g"],
    rgb_stretch=0.5,
    with_point_source=True,
    with_source=True,
    with_deflector=True,
):
    multiband_image_selected_lens = {}

    for i, band in enumerate(bands):
        simulated_lens_image = simulate_image(
            lens_class=lens_class,
            band=band,
            num_pix=num_pix,
            coadd_years=coadd_years,
            add_noise=add_noise,
            observatory="LSST",
            with_point_source=with_point_source,
            with_source=with_source,
            with_deflector=with_deflector,
        )
        multiband_image_selected_lens[band] = simulated_lens_image

    rgb_image = rgb_image_from_image_list(
        image_list=[
            multiband_image_selected_lens[rgb_bands[0]],  
            multiband_image_selected_lens[rgb_bands[1]],  
            multiband_image_selected_lens[rgb_bands[2]],  
        ],
        stretch=rgb_stretch,
    )
    return multiband_image_selected_lens, rgb_image


def plot_montage(
    nonlenses,
    number_to_plot=100,
    num_cols=10,
    bands=["g", "r", "i", "z", "y"],
    rgb_bands=["i", "r", "g"],
    rgb_stretch=0.5,
    with_point_source=True,
    with_extended_source=True,
    with_deflector=True,
    num_pix=41,
    coadd_years=10,
    add_noise=True,
    plot_title=None,
    random_seed=None,
):
    if len(nonlenses) > number_to_plot:
        np.random.seed(random_seed)
        random_idxs = np.random.choice(len(nonlenses), size=number_to_plot, replace=False)
        nonlenses_to_plot = [nonlenses[i] for i in random_idxs]
    else:
        nonlenses_to_plot = nonlenses

    all_rgb_images = []
    for lens_class in nonlenses_to_plot:
        _, rgb_image = make_multiband_images_and_rgb_image(
            lens_class, bands=bands, num_pix=num_pix, coadd_years=coadd_years,
            add_noise=add_noise, with_point_source=with_point_source,
            with_source=with_extended_source, with_deflector=with_deflector,
            rgb_bands=rgb_bands, rgb_stretch=rgb_stretch,
        )
        all_rgb_images.append(rgb_image)

    num_images = len(all_rgb_images)
    num_rows = int(np.ceil(num_images / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(all_rgb_images[i], origin="lower")
            ax.axis("off")
        else:
            ax.remove()
    if plot_title:
        plt.suptitle(plot_title, fontsize=16)
        plt.subplots_adjust(top=0.95)
    fig.tight_layout()
    return fig

def extract_lens_properties(lens_objects, all_bands=None, max_num_images=5,):
    """
    Extracts properties from lens objects using the Galaxy-Galaxy Lensing Challenge 
    naming convention. Generates IDs and random coordinates.

    Parameters
    ----------
    lens_objects : list
        List of lens system objects.
    all_bands : list of str, optional
        Bands for magnitudes. Defaults to ['g', 'r', 'i', 'z', 'y'].
    max_num_images : int, optional
        Max images for column allocation (default 5).

    Returns
    -------
    astropy.table.Table
        Table with standardized column names (e.g., 'vel_disp', 'Rein', 'ell_m').
    """
    
    if all_bands is None:
        all_bands = ['g', 'r', 'i', 'z', 'y']

    # -------------------------------------------------------------------------
    # 1. Initialize Dictionary with Standardized Names
    # -------------------------------------------------------------------------
    table_dict = {
        # --- Metadata ---
        "Lens ID": [],
        "RA": [],
        "Dec": [],
        "num_images": [],

        # --- Deflector Mass (Lens Galaxy) ---
        "zlens": [],          # Redshift of deflector
        "vel_disp": [],       # Velocity dispersion (km/s)
        "ell_m": [],          # Ellipticity of mass
        "ell_m_PA": [],       # PA of mass ellipticity
        "sh": [],             # Shear amplitude
        "sh_PA": [],          # Shear PA
        "Rein": [],           # Einstein radius (arcsec)
        
        # --- Deflector Light (Lens Galaxy) ---
        "ell_l": [],          # Ellipticity of light
        "ell_l_PA": [],       # PA of light
        "Reff_l": [],         # Effective radius (arcsec)
        "n_l_sers": [],       # Sersic index
        # (mag_lens_{band} added dynamically below)

        # --- Source Properties (Extended Host) ---
        "zsrc": [],           # Redshift of source
        "srcx": [],           # x-position (arcsec)
        "srcy": [],           # y-position (arcsec)
        "ell_s": [],          # Ellipticity of source
        "ell_s_PA": [],       # PA of source
        "Reff_s": [],         # Effective radius (arcsec)
        "n_s_sers": [],       # Sersic index
        # (mag_src_{band} added dynamically below)

        # --- Point Source / AGN (Adapted Style) ---
        "psx": [],            # x-position of AGN
        "psy": [],            # y-position of AGN
        "bh_mass_exp": [],    # Black hole mass exponent
        "edd_ratio": [],      # Eddington ratio
        "dt_max": [],         # Max time delay
    }

    # --- Dynamic Band Columns ---
    for band in all_bands:
        table_dict[f"mag_lens_{band}"] = [] # Deflector magnitude
        table_dict[f"mag_src_{band}"] = []  # Unlensed Source (Host) magnitude
        table_dict[f"mag_ps_{band}"] = []   # Unlensed Point Source magnitude

    # --- Dynamic Image Columns (Quasar Images) ---
    for img_num in range(1, max_num_images + 1):
        table_dict[f"dt_days_{img_num}"] = []  # Time delays
        table_dict[f"mu_{img_num}"] = []       # Magnification (mu)
        table_dict[f"x_img_{img_num}"] = []    # Image X position
        table_dict[f"y_img_{img_num}"] = []    # Image Y position
        
        for band in all_bands:
            table_dict[f"mag_ps_img_{band}_{img_num}"] = [] # Lensed PS mag per image

    # -------------------------------------------------------------------------
    # 2. Populate Data
    # -------------------------------------------------------------------------
    for i, lens_system in enumerate(tqdm(lens_objects, desc="Extracting lens properties")):
        
        # --- Metadata Generation ---
        # Format: D1_L000YYYYY where YYYYY is index (padded to 5 digits)
        obj_id = f"D1_L000{i:05d}"
        
        # Random RA and Dec in LSST footprint (RA: 0-360, Dec: -72 to +12)
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-72, 12)
        
        table_dict["Lens ID"].append(obj_id)
        table_dict["RA"].append(ra)
        table_dict["Dec"].append(dec)

        # --- Access Internal Classes ---
        source_index = 0 
        source_class = lens_system.source(source_index)
        deflector_class = lens_system.deflector
        ps_class = source_class._source._point_source
        es_class = source_class._source._extended_source

        # --- Basic Redshifts & Geometry ---
        table_dict["zsrc"].append(lens_system.source_redshift_list[source_index])
        table_dict["zlens"].append(lens_system.deflector_redshift)
        table_dict["Rein"].append(lens_system.einstein_radius[source_index])
        table_dict["num_images"].append(lens_system.image_number[source_index])

        # --- Deflector Mass Properties ---
        # Ellipticities & Angles
        e1_l, e2_l, e1_m, e2_m = lens_system.deflector_ellipticity()
        
        # Mass
        ell_m = np.sqrt(e1_m**2 + e2_m**2)
        phi_m = 0.5 * np.degrees(np.arctan2(e2_m, e1_m))
        table_dict["ell_m"].append(ell_m)
        table_dict["ell_m_PA"].append(phi_m)
        table_dict["vel_disp"].append(lens_system.deflector_velocity_dispersion())

        # Shear
        kappa_ext, g1, g2 = lens_system.los_linear_distortions
        sh = np.sqrt(g1**2 + g2**2)
        sh_pa = 0.5 * np.degrees(np.arctan2(g2, g1))
        table_dict["sh"].append(sh)
        table_dict["sh_PA"].append(sh_pa)

        # --- Deflector Light Properties ---
        ell_l = np.sqrt(e1_l**2 + e2_l**2)
        phi_l = 0.5 * np.degrees(np.arctan2(e2_l, e1_l))
        
        table_dict["ell_l"].append(ell_l)
        table_dict["ell_l_PA"].append(phi_l)
        table_dict["Reff_l"].append(lens_system.deflector.angular_size_light)
        
        # Safe fetch for Sersic index
        n_l = np.nan
        if hasattr(deflector_class._deflector, '_deflector_dict'):
            n_l = deflector_class._deflector._deflector_dict.get("n_sersic", np.nan)
        table_dict["n_l_sers"].append(n_l)

        # --- Source (Host) Properties ---
        srcx, srcy = es_class.extended_source_position
        table_dict["srcx"].append(srcx)
        table_dict["srcy"].append(srcy)
        table_dict["Reff_s"].append(es_class.angular_size)
        table_dict["n_s_sers"].append(es_class._n_sersic)
        
        e1_s, e2_s = es_class.ellipticity
        ell_s = np.sqrt(e1_s**2 + e2_s**2)
        phi_s = 0.5 * np.degrees(np.arctan2(e2_s, e1_s))
        table_dict["ell_s"].append(ell_s)
        table_dict["ell_s_PA"].append(phi_s)

        # --- Point Source / AGN Properties ---
        psx, psy = source_class.point_source_position
        table_dict["psx"].append(psx)
        table_dict["psy"].append(psy)
        table_dict["bh_mass_exp"].append(ps_class.source_dict.get("black_hole_mass_exponent", np.nan))
        table_dict["edd_ratio"].append(ps_class.source_dict.get("eddington_ratio", np.nan))

        # --- Time Delays & Mags ---
        arrival_times = lens_system.point_source_arrival_times()[source_index]
        if len(arrival_times) > 0:
            dt_max = np.max(arrival_times) - np.min(arrival_times)
        else:
            dt_max = np.nan
        table_dict["dt_max"].append(dt_max)

        # --- Band Loop ---
        for band in all_bands:
            # 1. Deflector Mag (mag_lens)
            table_dict[f"mag_lens_{band}"].append(lens_system.deflector_magnitude(band=band))
            
            # 2. Source Mag (mag_src - Unlensed Host)
            mag_src = lens_system.extended_source_magnitude(band=band, lensed=False)[source_index]
            table_dict[f"mag_src_{band}"].append(mag_src)

            # 3. PS Mag (mag_ps - Unlensed AGN)
            mag_ps = lens_system.point_source_magnitude(band=band, lensed=False)[source_index]
            table_dict[f"mag_ps_{band}"].append(mag_ps)

            # 4. Lensed PS Images
            lensed_mags = lens_system.point_source_magnitude(band=band, lensed=True)[source_index]
            
            for img_num in range(1, max_num_images + 1):
                col_name = f"mag_ps_img_{band}_{img_num}"
                val = lensed_mags[img_num-1] if img_num <= len(lensed_mags) else np.nan
                table_dict[col_name].append(val)

        # --- Image Positions & Magnifications ---
        ps_magnifications = lens_system.point_source_magnification()[source_index]
        x_img, y_img = lens_system.point_source_image_positions()[source_index]
        
        for img_num in range(1, max_num_images + 1):
            valid = (img_num <= lens_system.image_number[source_index])
            
            # Time Delays
            dt = arrival_times[img_num-1] if valid and img_num <= len(arrival_times) else np.nan
            table_dict[f"dt_days_{img_num}"].append(dt)

            # Magnification
            mu = ps_magnifications[img_num-1] if valid and img_num <= len(ps_magnifications) else np.nan
            table_dict[f"mu_{img_num}"].append(mu)

            # Positions
            ix = x_img[img_num-1] if valid and img_num <= len(x_img) else np.nan
            iy = y_img[img_num-1] if valid and img_num <= len(y_img) else np.nan
            table_dict[f"x_img_{img_num}"].append(ix)
            table_dict[f"y_img_{img_num}"].append(iy)

    return Table(table_dict)


def extract_non_lens_properties(non_lens_objects, all_bands=None, has_central_object=True, starting_index=0):
    """
    Extracts properties from a list of non-lens objects (structured identically to lens objects)
    following standardized naming conventions. Generates IDs and random coordinates.

    Parameters
    ----------
    non_lens_objects : list
        A list of non-lens objects. Assumed to have the same class structure as lens objects 
        (where the central object is represented by the 'deflector').
    all_bands : list of str, optional
        List of photometric bands to extract magnitudes for. Defaults to ['g', 'r', 'i', 'z', 'y'].
    has_central_object : bool, optional
        Flag indicating if the non-lens system has a central object (deflector). 
        If False, z_central and mag_lens_{band} will be set to np.nan. Defaults to True.
    starting_index : int, optional
        The starting index for generating object IDs. Defaults to 0.

    Returns
    -------
    astropy.table.Table
        A table containing Object ID, RA, Dec, z_central, and magnitudes.
    """
    
    if all_bands is None:
        all_bands = ['g', 'r', 'i', 'z', 'y']

    # -------------------------------------------------------------------------
    # Initialize table_dict
    # -------------------------------------------------------------------------
    table_dict = {
        "Object ID": [],
        "RA": [],
        "Dec": [],
        "z_central": [],
    }

    # Initialize Band-Dependent Keys
    for band in all_bands:
        table_dict[f"mag_lens_{band}"] = []

    # -------------------------------------------------------------------------
    # Populate properties
    # -------------------------------------------------------------------------
    for i, lens_system in enumerate(non_lens_objects):

        # --- Metadata Generation ---
        # Format: D1_N000YYYYY where YYYYY is the index
        obj_id = f"D1_N000{i + starting_index:05d}"
        
        # Random RA and Dec in LSST footprint (RA: 0-360, Dec: -72 to +12)
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-72, 12)

        table_dict["Object ID"].append(obj_id)
        table_dict["RA"].append(ra)
        table_dict["Dec"].append(dec)

        # --- Central Object Properties Extraction ---
        if has_central_object:
            # treating the deflector as the central object, which is what FalsePositivePop do in slsim
            z = lens_system.deflector_redshift
            val = z[0] if isinstance(z, (list, np.ndarray)) else z
            table_dict["z_central"].append(val)

            for band in all_bands:
                mag = lens_system.deflector_magnitude(band=band)
                table_dict[f"mag_lens_{band}"].append(mag)
        else:
            # If no central object, fill with NaNs
            table_dict["z_central"].append(np.nan)
            
            for band in all_bands:
                table_dict[f"mag_lens_{band}"].append(np.nan)

    # Create astropy table
    return Table(table_dict)