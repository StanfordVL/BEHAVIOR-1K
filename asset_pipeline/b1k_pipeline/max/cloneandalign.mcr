macroScript cloneandalign Buttontext: "Clone and Align" category: "SVL-Tools"
(
	
	On isEnabled do	--the script will not be available if no objects are selected
		(
			selection.count > 0
		)--end on isenabled
		
		On Execute do
		(
			undo on	--Undo the operation done using the script
			(
				local source_obj = pickobject()
				for target_obj in selection do
				(
					maxOps.cloneNodes source_obj cloneType:#instance newNodes:&new_obj #nodialog
					new_obj.transform = target_obj.transform
					new_obj.scale = target_obj.scale
				)
			)--end undo on
		)--end on execute
)--end macroscript