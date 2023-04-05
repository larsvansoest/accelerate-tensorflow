{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -Wno-orphans #-}

import Data.Array.Accelerate.AST.Operation
    ( expType,
      paramsIn',
      fromGrounds,
      PrimFun(..),
      Var(Var),
      PreArgs((:>:), ArgsNil),
      ExpVars,
      PreOpenAcc(Compute, Exec, Return, Alet),
      GroundR(GroundRscalar),
      Modifier(In, Mut, Out),
      PreOpenExp(..),
      PreOpenFun(Body, Lam),
      ArrayInstr(..),
      GroundVars,
      OperationAcc,
      Out,
      In,
      Arg(ArgFun, ArgArray) )
import Data.Array.Accelerate.AST.LeftHandSide
    ( LeftHandSide(LeftHandSideWildcard, LeftHandSideSingle) )
import Data.Array.Accelerate.Backend
    ( DesugarAcc(mkGenerate, mkPermute, mkMap) )
import Data.Array.Accelerate.Type
    ( Word8,
      scalarTypeWord8,
      scalarTypeInt,
      IntegralType(TypeInt),
      NumType(IntegralNumType, FloatingNumType),
      SingleType(NumSingleType),
      ScalarType(SingleScalarType) )
import Data.Array.Accelerate.Representation.Array
    ( Buffer, Buffers, ArrayR(ArrayR) )
import Data.Array.Accelerate.Representation.Type
    ( Distributes(reprIsSingle),
      Distribute,
      TupR(TupRpair, TupRunit, TupRsingle),
      TypeR )
import Data.Array.Accelerate.Trafo.Var
    ( declareVars, DeclareVars(DeclareVars) )
import Data.Array.Accelerate.AST.Idx ( Idx(ZeroIdx, SuccIdx) )
import Data.Array.Accelerate.Trafo.Operation.Substitution
    ( weakenArrayInstr, aletUnique, Sink(..) )
import Data.Array.Accelerate.AST.Environment
    ( mapEnv, prj', (.>), weakenId, push', type (:>), Env(Empty) )
import Data.Array.Accelerate.Representation.Ground ( buffersR )
import Data.Array.Accelerate.Trafo.Desugar ( desugarAlloc )
import Data.Array.Accelerate.Trafo.Exp.Substitution
    ( weakenVars, SinkExp(weakenE) )
import Data.Type.Equality ( type (:~:)(Refl) )
import Data.Array.Accelerate.Representation.Shape
    ( dim1, shapeType, ShapeR(..), DIM1 )
import Prelude hiding (exp)

import Data.Accelerate.TensorFlow.Type
import Data.Accelerate.TensorFlow.Operation

instance DesugarAcc TensorOp where
  mkMap (ArgFun (Lam lhs (Body body))) (ArgArray _ (ArrayR _ t) _ gvb) aOut =
    mkMapF (push' Empty (lhs, distributeBIdx t gvb)) body aOut
  mkMap _ _ _ = error "impossible"

  mkGenerate f (ArgArray _ (ArrayR sh t) gv gvb)
    | DeclareVars lhs w k       <- declareVars $ buffersR (TupRsingle scalarTypeInt)
    , DeclareVars lhs' w' k'    <- declareVars $ buffersR (shapeType sh)
    , DeclareVars lhs'' w'' k'' <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
    = -- 1) Create a Tensor of shape sh with only ones
      aletUnique lhs (desugarAlloc (ArrayR sh (TupRsingle scalarTypeInt)) (fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        (TConstant scalarTypeInt 1)
        (ArgArray Out (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars w gv) (k weakenId) :>: ArgsNil)
      ) $
      -- 2) Obtain a 1D array of indexes (tf.where returns list of indices pointing to values > 0)
      aletUnique lhs' (desugarAlloc (ArrayR sh (shapeType sh)) (weakenVars w $ fromGrounds gv)) $
      aletUnique lhs'' (Compute (ShapeSize sh (paramsIn' $ weakenVars (w' .> w) $ fromGrounds gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        (TWhere scalarTypeInt)
        (ArgArray In (ArrayR sh (TupRsingle scalarTypeInt)) (weakenVars (w'' .> w' .> w) gv) (k (w'' .> w')) :>:
         ArgArray Out (ArrayR dim1 (shapeType sh)) (TupRpair TupRunit (k'' weakenId)) (k' w'') :>:
         ArgsNil)
      )
      -- 3) Apply map on array of indices.
      (mkMap
        (weaken (w'' .> w' .> w) f)
        (ArgArray In (ArrayR dim1 (shapeType sh)) (TupRpair TupRunit (k'' weakenId)) (k' w''))
        (ArgArray Out (ArrayR dim1 t) (TupRpair TupRunit (k'' weakenId)) (weakenVars (w'' .> w' .> w) gvb)))

  mkPermute
    (ArgFun comb)
    (ArgArray _ (ArrayR sh' _) gv' gvb')
    perm
    (ArgArray _ (ArrayR sh t) gv gvb)
    | maybeSh'                     <- TupRpair (TupRsingle scalarTypeWord8) (TupRpair TupRunit (shapeType sh'))
    , DeclareVars lhs w k          <- declareVars $ buffersR maybeSh'
    , DeclareVars lhs' w' k'       <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
    , DeclareVars lhs'' w'' k''    <- declareVars $ buffersR (shapeType sh')
    , DeclareVars lhs''' w''' k''' <- declareVars $ buffersR t
    = -- 1) Create an array of maybeSh' with perm
      aletUnique lhs (desugarAlloc (ArrayR sh maybeSh') (fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkGenerate
        (weaken w perm)
        (ArgArray Out (ArrayR sh maybeSh') (weakenVars w gv) (k weakenId))
      ) $
      -- 2) To apply boolean mask with 1D arrays, we need to flatten. Calculate the dim1 size.
      aletUnique lhs' (Compute (ShapeSize sh (paramsIn' $ weakenVars w $ fromGrounds gv))) $
      -- 3) Get 1D array of (Just sh'), by applying a boolean mask with predicate isJust.
      aletUnique lhs'' (desugarAlloc (ArrayR sh (shapeType sh')) (fromGrounds (weakenVars (w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask (shapeType sh')
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' w'')) (isJust (TupRpair TupRunit (shapeType sh')) (k (w'' .> w'))))
        (fromJust (shapeType sh') (k (w'' .> w')))
        (k'' weakenId)
      ) $
      -- 3) Get 1D array of source indices (sh) with perm output (Just sh'), by applying a boolean mask with predicate isJust.
      aletUnique lhs''' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w'' .> w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask t
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' (w''' .> w''))) (isJust (TupRpair TupRunit (shapeType sh')) (k (w''' .> w'' .> w'))))
        (weakenVars (w''' .> w'' .> w' .> w) gvb)
        (k''' weakenId)
      ) $
      -- 4) Apply tf.tensor_scatter
      Exec (TTensorScatter scatterFun) (
        ArgArray Mut (ArrayR sh' t) (weakenVars (w''' .> w'' .> w' .> w) gv') (weakenVars (w''' .> w'' .> w' .> w) gvb') :>:
        ArgArray In (ArrayR sh (shapeType sh')) (weakenVars (w''' .> w'' .> w' .> w) gv) (k'' w''') :>:
        ArgArray In (ArrayR sh t) (weakenVars (w''' .> w'' .> w' .> w) gv) (k''' weakenId) :>:
        ArgsNil
      )
        where scatterFun = case comb of
                Lam (LeftHandSideSingle _) (Lam (LeftHandSideSingle _) (Body (PrimApp fun (Pair (Evar (Var _ (SuccIdx ZeroIdx))) (Evar (Var _ ZeroIdx)))))) -> case fun of
                  PrimAdd _ -> ScatterFunAdd
                  PrimSub _ -> ScatterFunMin
                  _         -> error "primfun not yet supported"
                _ -> error "complex combination for permute not supported"

booleanMask :: TypeR a -> Arg env (In DIM1 Word8) -> GroundVars env (Buffers a) -> GroundVars env (Buffers a) -> PreOpenAcc TensorOp env ()
booleanMask TupRunit _ _ gvb = Return gvb
booleanMask t@(TupRsingle s) aOut@(ArgArray _ _ gv _) gvbIn gvbOut = Exec (TBooleanMask s) (ArgArray In (ArrayR dim1 t) gv gvbIn :>: aOut :>: ArgArray Out (ArrayR dim1 t) gv gvbOut :>: ArgsNil)
booleanMask (TupRpair t1 t2) aOut (TupRpair gvbIn1 gvbIn2) (TupRpair gvbOut1 gvbOut2) = Alet (LeftHandSideWildcard TupRunit) TupRunit
 (booleanMask t1 aOut gvbIn1 gvbOut1)
 (booleanMask t2 aOut gvbIn2 gvbOut2)
booleanMask _ _ _ _ = error "impossible"

isJust :: TypeR a -> GroundVars env (Buffers (Word8, a)) -> GroundVars env (Buffers Word8)
isJust _ (TupRpair word8 _) = word8
isJust _ _ = error "impossible"

fromJust :: TypeR a -> GroundVars env (Buffers (Word8, ((), a))) -> GroundVars env (Buffers a)
fromJust _ (TupRpair _ (TupRpair _ a)) = a
fromJust _ _ = error "impossible"

mkMapF :: forall env env' sh t. BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' t
  -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapF _ (Const s e) aOut | OneOfDict <- tfAllDict s = Exec (TConstant s e) $ aOut :>: ArgsNil

mkMapF env (PrimApp f exp) aOut = mkMapPrimAppF f env exp aOut

mkMapF env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp1
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF
     (weakenEnv w env)
     (weakenArrayInstr w exp1)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   mkMapF
     (push' (weakenEnv w env) (elhs, distributeBIdx a (k weakenId)))
     (weakenArrayInstr w exp2)
     (weaken w aOut)

mkMapF env (Evar (Var s idx)) (ArgArray _ arrayR gv gvb@(TupRsingle (Var groundR _)))
  | Refl <- reprIsSingle @ScalarType @t @Buffer s
  , OneOfDict <- tfAllDict s
  = let (BIdx idx') = prj' idx env
        gvb'        = TupRsingle (Var groundR idx')
    in  Exec
          (TId s)
          (
            ArgArray In arrayR gv gvb' :>:
            ArgArray Out arrayR gv gvb :>:
            ArgsNil
          )

mkMapF env (Pair exp1 exp2) (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2))
 = Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF env exp1 (ArgArray Out (ArrayR sh t1) gv gvb1))
   (mkMapF env exp2 (ArgArray Out (ArrayR sh t2) gv gvb2))

mkMapF env (ToIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ shapeType sh
  = mkMapF env (Let lhs exp1 (Let lhs' (weakenE w exp2) (toIndex sh (k w') (k' weakenId)))) aOut

mkMapF env (ShapeSize sh exp) aOut
  | DeclareVars lhs _ k <- declareVars $ shapeType sh
  = mkMapF env (Let lhs exp (shapeSize sh (k weakenId))) aOut

mkMapF env (FromIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ TupRsingle scalarTypeInt
  = mkMapF env (Let lhs exp1 (Let lhs' (weakenE w exp2) (fromIndex sh (k w') (k' weakenId)))) aOut

mkMapF env (Cond cond exp1 exp2) (ArgArray _ (ArrayR sh t) gv gvb) 
  | DeclareVars lhs w k <- declareVars $ buffersR $ TupRsingle scalarTypeWord8
  , DeclareVars lhs' w' k' <- declareVars $ buffersR t
  , DeclareVars lhs'' w'' k'' <- declareVars $ buffersR t
  = aletUnique lhs (desugarAlloc (ArrayR sh (TupRsingle scalarTypeWord8)) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkMapF (weakenEnv w env) (weakenArrayInstr w cond) (ArgArray Out (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars w gv) (k weakenId))) $
    aletUnique lhs' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars w gv))) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkMapF (weakenEnv (w' .> w) env) (weakenArrayInstr (w' .> w) exp1) (ArgArray Out (ArrayR sh t) (weakenVars (w' .> w) gv) (k' weakenId))) $
    aletUnique lhs'' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w' .> w) gv))) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkMapF (weakenEnv (w'' .> w' .> w) env) (weakenArrayInstr (w'' .> w' .> w) exp2) (ArgArray Out (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId)))
    (Exec
      TCond
      (
        ArgArray In (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars (w'' .> w' .> w) gv) (k (w'' .> w')) :>:
        ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k' w'') :>:
        ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId) :>:
        ArgArray Out (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (weakenVars (w'' .> w' .> w) gvb) :>:
        ArgsNil
      )
    )

mkMapF _ Nil _ = Return TupRunit

mkMapF env (ArrayInstr (Index (Var groundR idx)) exp) (ArgArray _ (ArrayR sh t) gv gvb)
  | a <- expType exp
  , sh' <- shapeType sh
  , DeclareVars lhs w k <- declareVars $ buffersR a
  = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $ undefined

mkMapF env (ArrayInstr (Parameter (Var s idx)) exp) (ArgArray _ (ArrayR sh t) gv gvb) = undefined

mkMapF _ (Foreign _ _ _ _) _ = undefined
mkMapF _ (VecPack _ _) _ = undefined
mkMapF _ (VecUnpack _ _) _ = undefined
mkMapF _ (IndexSlice _ _ _) _ = undefined
mkMapF _ (IndexFull _ _ _) _ = undefined

mkMapF _ (Case _ _ _) _ = undefined

mkMapF _ (While _ _ _) _ = undefined -- see loopcond?
mkMapF _ (PrimConst _) _ = undefined

mkMapF _ (Undef _) _ = undefined
mkMapF _ (Coerce _ _ _) _ = undefined
mkMapF _ _ _ = error "impossible"

mkMapPrimAppF :: PrimFun (a -> t) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapPrimAppF (PrimAdd nt)  | OneOfDict <- tfNumDict nt = mkMapPrimAppF' $ TAdd (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimMul nt)  | OneOfDict <- tfNumDict nt = mkMapPrimAppF' $ TMul (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimSub nt)  | OneOfDict <- tfNumDict nt = mkMapPrimAppF' $ TSub (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimNeg nt)  | OneOfDict <- tfNumDict nt = mkMapPrimAppF' $ TNeg (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimAbs nt)  | OneOfDict <- tfNumDict nt = mkMapPrimAppF' $ TAbs (SingleScalarType (NumSingleType nt))
mkMapPrimAppF (PrimSig nt)  | OneOfDict <- tfNumDict nt = mkMapPrimAppF' $ TSign (SingleScalarType (NumSingleType nt))

mkMapPrimAppF (PrimQuot it) | OneOfDict <- tfNumDict (IntegralNumType it) = mkMapPrimAppF' $ TTruncateDiv (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimRem it)  | OneOfDict <- tfNumDict (IntegralNumType it) = mkMapPrimAppF' $ TTruncateMod (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimQuotRem it) = undefined
mkMapPrimAppF (PrimIDiv it) | OneOfDict <- tfNumDict (IntegralNumType it) = mkMapPrimAppF' $ TRealDiv (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimMod it)  | OneOfDict <- tfNumDict (IntegralNumType it) = mkMapPrimAppF' $ TTruncateMod (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimDivMod it) = undefined
 
mkMapPrimAppF (PrimBAnd it) | OneOfDict <- tfIntDict it = mkMapPrimAppF' $ TBitwiseAnd (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimBOr it)  | OneOfDict <- tfIntDict it = mkMapPrimAppF' $ TBitwiseOr (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimBXor it) | OneOfDict <- tfIntDict it = mkMapPrimAppF' $ TBitwiseXor (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimBNot it) | OneOfDict <- tfIntDict it = mkMapPrimAppF' $ TInvert (SingleScalarType (NumSingleType (IntegralNumType it)))
mkMapPrimAppF (PrimBShiftL it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimBShiftR it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimBRotateL it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimBRotateR it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimPopCount it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimCountLeadingZeros it) | OneOfDict <- tfIntDict it = undefined
mkMapPrimAppF (PrimCountTrailingZeros it) | OneOfDict <- tfIntDict it = undefined

mkMapPrimAppF (PrimFDiv ft) | OneOfDict <- tfNumDict (FloatingNumType ft) = mkMapPrimAppF' $ TRealDiv (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimRecip ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TReciprocal (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimSin ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TSin (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimCos ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TCos (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimTan ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TTan (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAsin ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAsin (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAcos ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAcos (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAtan ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAtan (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimSinh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TSinh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimCosh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TCosh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimTanh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TTanh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAsinh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAsinh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAcosh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAcosh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimAtanh ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAtanh (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimExpFloating ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TExp (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimSqrt ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TSqrt (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimLog ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TLog (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimFPow ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TPow (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimLogBase ft) | OneOfDict <- tfFloatDict ft = undefined -- mkMapPrimAppF' $ TLog1p (SingleScalarType (NumSingleType (FloatingNumType ft)))

mkMapPrimAppF (PrimTruncate ta tb) = undefined
mkMapPrimAppF (PrimRound ta tb) = undefined
mkMapPrimAppF (PrimFloor ta tb) = undefined
mkMapPrimAppF (PrimCeiling ta tb) = undefined

mkMapPrimAppF (PrimAtan2 ft) | OneOfDict <- tfFloatDict ft = mkMapPrimAppF' $ TAtan2 (SingleScalarType (NumSingleType (FloatingNumType ft)))
mkMapPrimAppF (PrimIsNaN _) = undefined
mkMapPrimAppF (PrimIsInfinite _) = undefined

mkMapPrimAppF (PrimLt st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TLess (SingleScalarType st)
mkMapPrimAppF (PrimGt st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TGreater (SingleScalarType st)
mkMapPrimAppF (PrimLtEq st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TLessEqual (SingleScalarType st)
mkMapPrimAppF (PrimGtEq st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TGreaterEqual (SingleScalarType st)
mkMapPrimAppF (PrimEq st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TEqual (SingleScalarType st)
mkMapPrimAppF (PrimNEq st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TNotEqual (SingleScalarType st)
mkMapPrimAppF (PrimMax st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TMaximum (SingleScalarType st)
mkMapPrimAppF (PrimMin st) | OneOfDict <- tfOrdDict st = mkMapPrimAppF' $ TMinimum (SingleScalarType st)

mkMapPrimAppF PrimLAnd = mkMapPrimAppF' TLogicalAnd
mkMapPrimAppF PrimLOr = mkMapPrimAppF' TLogicalOr
mkMapPrimAppF PrimLNot = mkMapPrimAppF' TLogicalNot

mkMapPrimAppF (PrimFromIntegral it nt) 
  | TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType it)))
  , TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType nt))
  = mkMapPrimAppF' $ TCast (SingleScalarType (NumSingleType (IntegralNumType it))) (SingleScalarType (NumSingleType nt))

mkMapPrimAppF (PrimToFloating _ _) = undefined

mkMapPrimAppF' :: TensorOp (In sh a -> Out sh b -> ()) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh b) -> OperationAcc TensorOp env ()
mkMapPrimAppF' op env exp aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF -- flatten higher-order expression
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   Exec -- apply method to the result
    op
    (
      ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
      weaken w aOut :>:
      ArgsNil
    )

shapeSize :: ShapeR sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
shapeSize ShapeRz TupRunit                                = Const scalarTypeInt 1
shapeSize (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz))
  = PrimApp (PrimMul (IntegralNumType TypeInt))
      (Pair
        (shapeSize shr sh)
        (Evar sz))
shapeSize _ _                                             = error "impossible"

toIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
toIndex ShapeRz TupRunit TupRunit = Const scalarTypeInt 0
toIndex (ShapeRsnoc shr) (TupRpair sh _) (TupRpair ix (TupRsingle i))
  = PrimApp (PrimAdd (IntegralNumType TypeInt))
      (Pair
        (toIndex shr sh ix)
        (PrimApp (PrimMul (IntegralNumType TypeInt))
          (Pair
            (shapeSize shr sh)
            (Evar i))))
toIndex _ _ _                     = error "impossible"

fromIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' Int -> PreOpenExp (ArrayInstr env) env' sh
fromIndex ShapeRz TupRunit _                                = Nil
fromIndex (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz)) (TupRsingle i)
  | DeclareVars lhs w k <- declareVars $ TupRsingle scalarTypeInt
  = Pair
    (Let 
      lhs 
      (PrimApp (PrimQuot TypeInt) (Pair (Evar i) (Evar sz))) 
      (fromIndex shr (weakenVars w sh) (k weakenId)))
    (case shr of
      ShapeRz -> Evar i
      _       -> PrimApp (PrimRem TypeInt) (Pair (Evar i) (Evar sz))
    )

fromIndex _ _ _              = error "impossible"

newtype BufferIdx benv a = BIdx (Idx benv (Buffer a))

instance Sink BufferIdx where
  weaken w (BIdx idx) = BIdx (weaken w idx)

type BufferEnv benv env = Env (BufferIdx benv) env

weakenEnv :: Sink f => (env1 :> env') -> Env (f env1) env2 -> Env (f env') env2
weakenEnv w = mapEnv (weaken w)

distributeBIdx :: forall env s. TypeR s -> GroundVars env (Buffers s) -> Distribute (BufferIdx env) s
distributeBIdx TupRunit _ = ()
distributeBIdx (TupRsingle s) (TupRsingle (Var _ idx))
  | Refl <- reprIsSingle @ScalarType @s @(BufferIdx env) s
  , Refl <- reprIsSingle @ScalarType @s @Buffer s
  = BIdx idx
distributeBIdx (TupRpair l1 l2) (TupRpair r1 r2) = (distributeBIdx l1 r1, distributeBIdx l2 r2)
distributeBIdx _ _ = error "impossible"
